import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorT5Full(nn.Module):
    """
    A predictor utilizing the full T5 encoder to predict embeddings for masked tokens.
    """

    def __init__(self, t5_encoder: nn.Module, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.t5 = t5_encoder
        self.d = d_model
        self.do = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        h_ctx_full: torch.Tensor,  # [L, D]
        idx_masked: torch.Tensor,  # [M]
        key_padding_mask: Optional[torch.Tensor] = None,  # [L] True = mask
    ) -> torch.Tensor:
        dev = h_ctx_full.device
        x = self.ln_in(h_ctx_full).unsqueeze(0)  # [1, L, D]
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).long().unsqueeze(0)  # [1, L]
        else:
            attn_mask = torch.ones((1, x.size(1)), dtype=torch.long, device=dev)

        with torch.enable_grad():
            out = self.t5(
                inputs_embeds=x, attention_mask=attn_mask
            ).last_hidden_state  # [1, L, D]
        h = self.ln_out(self.do(out[0]))  # [L, D]
        return h.index_select(0, idx_masked)  # [M, D]


# ------------------------------
# Predictor (Lite) + Hint annealing
# ------------------------------


class ResidualMLPBlock(nn.Module):
    """
    A residual Multi-Layer Perceptron block with Gated Linear Units (GLU).
    """

    def __init__(self, d_model: int, expansion: float = 1.5, dropout: float = 0.2):
        super().__init__()
        inner = int(expansion * d_model)
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, inner * 2)  # -> [inner, inner]
        self.act = nn.Sigmoid()  # For GEGLU: use F.gelu instead of Sigmoid
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner, d_model)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln(x)
        a, g = self.fc1(z).chunk(2, dim=-1)
        z = a * self.act(g)  # GLU
        z = self.drop(z)
        z = self.fc2(z)
        return x + z


class PredictorMLP(nn.Module):
    """
    A stack of ResidualMLPBlocks used as the projection head for the predictors.
    """

    def __init__(
        self, d_model: int, depth: int = 1, expansion: float = 1.5, dropout: float = 0.2
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model, expansion, dropout) for _ in range(depth)]
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)


class PredictorMicroAttn(nn.Module):
    """
    A lightweight predictor using adaptive micro-attention (local window attention)
    to predict masked tokens efficiently.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        window: Optional[int] = None,
        dropout: float = 0.2,
        min_keys: int = 24,
        base_window: int = 8,
    ):
        super().__init__()
        self.window = window  # max radius (None / <0 implies global)
        self.base_window = base_window  # small starting radius (e.g., 8)
        self.min_keys = min_keys  # minimum number of valid keys required
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.mlp = PredictorMLP(d_model, depth=1, expansion=1.5, dropout=dropout)

    def forward(
        self,
        h_ctx_full: torch.Tensor,
        idx_masked: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        force_visible_indices: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        L, D = h_ctx_full.shape
        q = self.ln_q(h_ctx_full.index_select(0, idx_masked)).unsqueeze(1)  # [M, 1, D]

        # ---- Global Mode ----
        if (self.window is None) or (isinstance(self.window, int) and self.window < 0):
            kv = self.ln_kv(h_ctx_full).unsqueeze(0).expand(q.size(0), L, D)
            kpm = (
                key_padding_mask.unsqueeze(0).expand(q.size(0), L)
                if key_padding_mask is not None
                else None
            )
            if kpm is not None and force_visible_indices is not None:
                for row, idx_force in enumerate(force_visible_indices):
                    if idx_force is None or idx_force.numel() == 0:
                        continue
                    idx_force = idx_force.to(device=h_ctx_full.device, dtype=torch.long)
                    idx_force = idx_force[(idx_force >= 0) & (idx_force < L)]
                    if idx_force.numel() > 0:
                        kpm[row, idx_force] = False
            # Guard: prevent rows from being entirely masked
            if kpm is not None:
                all_masked = kpm.all(dim=1)
                if all_masked.any():
                    idx = torch.nonzero(all_masked, as_tuple=False).squeeze(-1)
                    kpm[idx, idx_masked[idx]] = (
                        False  # unlock self-key using actual indices
                    )
            attn_out, _ = self.mha(q, kv, kv, key_padding_mask=kpm)
            return self.mlp(attn_out.squeeze(1))

        # ---- Adaptive Window Mode ----
        Wmax = int(self.window)
        W0 = min(int(self.base_window), Wmax)
        kpm_global = key_padding_mask  # [L] bool or None

        neighborhoods, masks, centers = [], [], []
        idx_masked_list = idx_masked.tolist()
        for q_idx, i in enumerate(idx_masked_list):
            idx_force = None
            if force_visible_indices is not None and q_idx < len(force_visible_indices):
                idx_force = force_visible_indices[q_idx]

            r = W0
            while True:
                left = max(0, i - r)
                right = min(L, i + r + 1)
                base_idx = torch.arange(left, right, device=h_ctx_full.device)

                if idx_force is not None and idx_force.numel() > 0:
                    idx_force = idx_force.to(device=h_ctx_full.device, dtype=torch.long)
                    idx_force = idx_force[(idx_force >= 0) & (idx_force < L)]
                    if idx_force.numel() > 0:
                        merged = torch.unique(
                            torch.cat(
                                [
                                    base_idx,
                                    idx_force,
                                    torch.tensor([i], device=h_ctx_full.device),
                                ]
                            )
                        )
                    else:
                        merged = base_idx
                else:
                    merged = base_idx

                merged, _ = torch.sort(merged)
                seg = h_ctx_full.index_select(0, merged)  # [t, D]
                t = seg.size(0)
                m = torch.zeros(t, dtype=torch.bool, device=h_ctx_full.device)
                if kpm_global is not None:
                    m |= kpm_global.index_select(0, merged)

                # Force-include selected relational nodes in the visible set.
                if idx_force is not None and idx_force.numel() > 0:
                    is_force = torch.isin(merged, idx_force)
                    if is_force.any():
                        m[is_force] = False

                # Number of valid keys in the current window
                valid = t - int(m.sum().item())
                # Stop when enough keys are found, or boundaries are hit and r=Wmax
                if (
                    (valid >= self.min_keys)
                    or ((left == 0 and right == L) and r == Wmax)
                    or (r >= Wmax)
                ):
                    neighborhoods.append(seg)
                    masks.append(m)
                    pos = torch.nonzero(merged == i, as_tuple=False)
                    centers.append(
                        int(pos[0].item()) if pos.numel() > 0 else 0
                    )  # position of the self-key within the window
                    break
                r = min(r + max(2, W0 // 2), Wmax)  # gradually expand, clamped by Wmax

        T_max = max(n.shape[0] for n in neighborhoods) if neighborhoods else 1
        kv = h_ctx_full.new_zeros((len(neighborhoods), T_max, D))
        kpm = torch.ones(
            (len(neighborhoods), T_max), dtype=torch.bool, device=h_ctx_full.device
        )
        for b, (seg, m, c) in enumerate(zip(neighborhoods, masks, centers)):
            t = seg.shape[0]
            kv[b, :t] = seg
            kpm[b, :t] = m
            # Guard: if the window is fully masked -> unmask the self-key
            if kpm[b, :t].all():
                kpm[b, c] = False

        kv = self.ln_kv(kv)
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=kpm)
        return self.mlp(attn_out.squeeze(1))


class PredictorSlidingWindowAttn(nn.Module):
    """
    A predictor using an efficient chunk-based sliding window attention mechanism
    for aggregating local context.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        window_overlap: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.W = window_overlap
        self.d_head = d_model // nhead

        self.ln_in = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Reuse the existing MLP block in the architecture
        self.mlp = PredictorMLP(d_model, depth=1, expansion=1.5, dropout=dropout)

    def _chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Based on the overlapping chunking algorithm: splits the sequence into chunks with overlap.
        hidden_states: [B, Seq, H, D_head]
        Returns: [B, n_chunks, 2W, H, D_head]
        """
        B, Seq, H, D = hidden_states.shape
        n_chunks = (Seq // self.W) - 1

        chunk_size = [B, n_chunks, self.W * 2, H, D]
        overlapping_chunks = torch.empty(
            chunk_size, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for chunk in range(n_chunks):
            overlapping_chunks[:, chunk, :, :, :] = hidden_states[
                :, chunk * self.W : chunk * self.W + 2 * self.W, :, :
            ]
        return overlapping_chunks

    def forward(
        self,
        h_ctx_full: torch.Tensor,
        idx_masked: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # h_ctx_full: [L, d_model]
        L, D = h_ctx_full.shape
        x = self.ln_in(h_ctx_full).unsqueeze(0)  # [1, L, D]

        # 1. Calculate padding to ensure L forms valid chunks
        # Target length (target_L) must be at least 2W and a multiple of W
        target_L = max(2 * self.W, math.ceil(L / self.W) * self.W)
        pad_len = target_L - L

        # 2. Pad empty values into the sequence and mask
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
            if key_padding_mask is not None:
                kpm_pad = F.pad(key_padding_mask, (0, pad_len), value=True)
            else:
                kpm_pad = torch.zeros(target_L, dtype=torch.bool, device=x.device)
                kpm_pad[L:] = True
        else:
            x_pad = x
            kpm_pad = (
                key_padding_mask
                if key_padding_mask is not None
                else torch.zeros(L, dtype=torch.bool, device=x.device)
            )

        # 3. Project Q, K, V
        qkv = self.qkv_proj(x_pad)  # [1, target_L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads: [1, target_L, H, d_head]
        q = q.view(1, target_L, self.nhead, self.d_head)
        k = k.view(1, target_L, self.nhead, self.d_head)
        v = v.view(1, target_L, self.nhead, self.d_head)

        # 4. Apply chunking
        q_c = self._chunk(q)
        k_c = self._chunk(k)
        v_c = self._chunk(v)

        # Chunk the Key Padding Mask as well
        n_chunks = (target_L // self.W) - 1
        kpm_c = torch.empty(
            (1, n_chunks, 2 * self.W), device=kpm_pad.device, dtype=torch.bool
        )
        for i in range(n_chunks):
            kpm_c[:, i, :] = kpm_pad[None, i * self.W : i * self.W + 2 * self.W]

        # 5. EINSUM ATTENTION SCORES
        # q_c, k_c shape: [b, c, x, h, d]. Where x, y are the chunk lengths (2W)
        scores = torch.einsum("bcxhd,bcyhd->bcxhy", q_c, k_c) / math.sqrt(self.d_head)

        # Apply mask
        mask = kpm_c.view(1, n_chunks, 1, 1, 2 * self.W)
        # Safety guard: Prevent NaN errors when an entire chunk is masked (all -inf)
        all_masked = mask.all(dim=-1, keepdim=True)
        mask = mask.masked_fill(all_masked, False)

        scores.masked_fill_(mask, float("-inf"))
        attn_probs = self.attn_drop(F.softmax(scores, dim=-1))

        # 6. EINSUM CONTEXT
        # attn_probs: [b, c, x, h, y], v_c: [b, c, y, h, d]
        out_c = torch.einsum("bcxhy,bcyhd->bcxhd", attn_probs, v_c)  # [1, c, 2W, h, d]

        # 7. Reassemble chunks and average the overlapping regions
        out_pad = torch.zeros(
            (1, target_L, self.nhead, self.d_head), device=x.device, dtype=x.dtype
        )
        counts = torch.zeros((1, target_L, 1, 1), device=x.device, dtype=x.dtype)

        for i in range(n_chunks):
            out_pad[:, i * self.W : i * self.W + 2 * self.W, :, :] += out_c[
                :, i, :, :, :
            ]
            counts[:, i * self.W : i * self.W + 2 * self.W, :, :] += 1.0

        out_pad = out_pad / counts  # Average the overlaps

        # 8. Merge heads and project output
        out_pad = out_pad.view(1, target_L, self.d_model)
        out = self.out_proj(out_pad)

        # Remove padding at the end and squeeze the batch dimension
        out = out[:, :L, :].squeeze(0)  # Back to [L, D]

        # 9. Select only the masked tokens to pass through the MLP (Extreme FLOPs optimization)
        out_masked = out.index_select(0, idx_masked)
        h_masked = h_ctx_full.index_select(0, idx_masked)

        z = h_masked + self.resid_drop(out_masked)
        return self.mlp(z)
