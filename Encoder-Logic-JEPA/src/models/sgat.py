from dataclasses import dataclass
import re
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

# ===================== Bias builder =====================


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """
    Calculate the length of the common prefix between two lists of integers.
    """
    n = min(len(a), len(b))
    k = 0
    for i in range(n):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k


def build_structural_prior(
    item: Dict,
    max_clip: float = 6.0,
    use_csl: bool = True,
    use_lg: bool = True,
    use_nlb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a channelized structural prior tensor P of shape [L, L, C],
    where C=3 represents the channels: [CSL, LG, NLB].
    Only channels with their respective flags set to True are computed;
    otherwise, they will remain as zero matrices.
    """
    tokens: List[Tuple[str, int]] = item["tokens"]
    seg_meta: List[Tuple[int, int]] = item["seg_meta"]
    L = len(tokens)

    # Extract paths and calculate depths (prefer new keys, keep old-key fallback)
    paths_list = (
        item.get("value_paths", [])
        or item.get("path", [])
        or item.get("type_paths", [])
        or item.get("leaf", [])
    )
    mp = {p["current_id"]: (p.get("path_ids") or []) for p in paths_list}

    paths_per_tok: List[List[int]] = []
    depths: List[int] = []
    for tok, gid in tokens:
        path_ids = mp.get(gid, [])
        d = max(len(path_ids) - 1, 0)
        paths_per_tok.append(path_ids)
        depths.append(d)

    depths_float = torch.tensor(depths, dtype=torch.float32)
    depths_t = depths_float.to(dtype=torch.long)

    # 1. CSL - Common Subtree Length (Vectorized)
    max_len = max((len(p) for p in paths_per_tok), default=0)

    if use_csl:
        if max_len > 0:
            P = torch.full((L, max_len), -1, dtype=torch.long)
            for i, p in enumerate(paths_per_tok):
                if p:
                    P[i, : len(p)] = torch.tensor(p, dtype=torch.long)

            # matches: [L, L, max_len]
            matches = (P.unsqueeze(1) == P.unsqueeze(0)) & (P.unsqueeze(1) != -1)
            # cumprod effectively finds continuous prefix matches from root
            prefix_mask = matches.int().cumprod(dim=-1)
            csl_raw = prefix_mask.sum(dim=-1).float()
            CSL = (csl_raw - 1.0).clamp_(min=0.0, max=max_clip)
        else:
            CSL = torch.zeros((L, L), dtype=torch.float32)
    else:
        CSL = torch.zeros((L, L), dtype=torch.float32)

    # 2. LG - Level Gap
    LG = torch.zeros((L, L), dtype=torch.float32)
    if use_lg:
        LG = (depths_float[:, None] - depths_float[None, :]).abs().clamp_(0, max_clip)

    # 3. NLB - Lexical Bridge
    NLB = torch.zeros((L, L), dtype=torch.float32)
    if use_nlb:
        mod_ids = [m for _, m in seg_meta]
        toks_norm = [t.lower() for (t, _) in tokens]
        is_bracket = [1 if t in {"(", ")"} else 0 for t in toks_norm]

        nl_idx = [i for i in range(L) if (mod_ids[i] == 0 and not is_bracket[i])]
        fol_idx = [j for j in range(L) if (mod_ids[j] == 1 and not is_bracket[j])]

        index: Dict[str, List[int]] = {}
        for j in fol_idx:
            fol = toks_norm[j]
            parts = [p for p in re.split(r"[_\W]+", fol) if p]
            for p in parts:
                index.setdefault(p, []).append(j)

        for i in nl_idx:
            j_list = index.get(toks_norm[i], None)
            if j_list:
                NLB[i, j_list] = 1.0
                NLB[j_list, i] = 1.0

        s_csl = CSL.pow(2).mean().sqrt()
        s_lg = LG.pow(2).mean().sqrt()
        s_nlb = NLB.pow(2).mean().sqrt()

        target = 0.5 * (s_csl + s_lg)
        scale = (target / (s_nlb + 1e-8)).clamp(max=max_clip)
        NLB = (NLB * scale).clamp_(0, max_clip)

    B_init = torch.stack([CSL, LG, NLB], dim=-1)  # [L, L, 3]
    return B_init, depths_t


# ===================== Structural Transformer =====================


@dataclass
class SGATConfig:
    d_model: int = 768
    nhead: int = 4
    num_layers: int = 1
    dim_ff: int = 1024
    dropout: float = 0.2
    bias_scale_init: float = 1.0
    bias_channels: int = 3

    # Control flags for Ablation Study
    use_csl: bool = True
    use_lg: bool = True
    use_nlb: bool = True


class SGATLayer(nn.Module):
    """
    A single layer of the Structure-Guided Attention Transformer (SGAT).
    Applies multi-head attention with structural bias, followed by a feed-forward network.
    """

    def __init__(self, cfg: SGATConfig):
        super().__init__()
        self.cfg = cfg
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        Forward pass for the SGAT layer.

        Args:
            x: Input tensor of shape [B, L, d_model].
            attn_bias: Structural bias tensor added to the attention scores.
            key_padding_mask: Optional mask for padded tokens.
            need_weights: Whether to return the attention weights.
        """
        B, L, _ = x.shape
        bias = attn_bias.to(dtype=x.dtype)
        bias = (
            bias.unsqueeze(1)
            .expand(B, self.cfg.nhead, L, L)
            .reshape(B * self.cfg.nhead, L, L)
        )

        attn_out, attn_w = self.mha(
            x,
            x,
            x,
            attn_mask=bias,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))

        if need_weights:
            if attn_w.dim() == 3:
                attn_w = attn_w.view(B, self.cfg.nhead, L, L)
            return x, attn_w
        return x, None


class SGAT(nn.Module):
    """
    Structure-Guided Attention Transformer (SGAT).
    Injects structural priors (CSL, LG, NLB) into the self-attention mechanism
    as learnable bias matrices.
    """

    def __init__(self, cfg: SGATConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([SGATLayer(cfg) for _ in range(cfg.num_layers)])

        self.gamma_c = nn.Parameter(torch.zeros(cfg.bias_channels))
        with torch.no_grad():
            if cfg.bias_channels >= 1 and cfg.use_csl:
                self.gamma_c[0] = 1.0  # +CSL
            if cfg.bias_channels >= 2 and cfg.use_lg:
                self.gamma_c[1] = -1.0  # -LG
            if cfg.bias_channels >= 3 and cfg.use_nlb:
                self.gamma_c[2] = 1.0  # +NLB

        # Hard ablation mask: disabled channels are always zero-contribution.
        # This guarantees strict on/off behavior even when optimizer updates gamma_c.
        channel_mask = torch.ones(cfg.bias_channels, dtype=torch.float32)
        if cfg.bias_channels >= 1 and not cfg.use_csl:
            channel_mask[0] = 0.0
        if cfg.bias_channels >= 2 and not cfg.use_lg:
            channel_mask[1] = 0.0
        if cfg.bias_channels >= 3 and not cfg.use_nlb:
            channel_mask[2] = 0.0
        self.register_buffer("gamma_channel_mask", channel_mask)

        self.b0 = nn.Parameter(torch.zeros(1))
        self.last_attn: List[torch.Tensor] = []
        self.record_attn: bool = True

    @property
    def bias_channels(self) -> int:
        return self.cfg.bias_channels

    def _make_attn_bias(self, raw_bias: torch.Tensor) -> torch.Tensor:
        """
        Combine the raw channelized bias tensor into a single attention bias matrix
        using learnable channel weights (gamma_c) and a global offset (b0).
        """
        gamma_eff = self.gamma_c * self.gamma_channel_mask.to(self.gamma_c.dtype)
        bias = torch.tensordot(raw_bias, gamma_eff, dims=([3], [0]))
        bias = bias + self.b0
        return bias

    def forward(
        self,
        x: torch.Tensor,
        raw_bias: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all SGAT layers.
        Combines the raw structural bias into an attention bias and applies it to the input.
        """
        self.last_attn = []
        attn_bias = self._make_attn_bias(raw_bias)

        need_w = bool(self.record_attn)
        for layer in self.layers:
            x, attn_w = layer(
                x,
                attn_bias,
                key_padding_mask=key_padding_mask,
                need_weights=need_w,
            )
            if attn_w is not None:
                self.last_attn.append(attn_w.mean(dim=1))
        return x
