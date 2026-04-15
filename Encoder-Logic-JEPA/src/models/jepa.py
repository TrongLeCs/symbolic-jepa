from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from src.models.predictor import (
    PredictorSlidingWindowAttn,
    PredictorT5Full,
    PredictorMicroAttn,
)


@torch.no_grad()
def ema_update(target: nn.Module, context: nn.Module, m: float = 0.996):
    """
    Update target network parameters using an Exponential Moving Average (EMA)
    of the context network parameters.
    """
    for p_t, p_o in zip(target.parameters(), context.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))


class LogicJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) for NL↔FOL (single-stream):
      - context: Encoder (processes masked inputs, supports batched forward).
      - target: Encoder (processes full inputs, stop-gradient, EMA updated; supports batched forward).
      - predictor: Predicts the embeddings of masked tokens from the context (Lightweight).
      - loss: Main MSE loss + (optional) Lexical Bridge Alignment (LBA) loss.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        d_model: int,
        lambda_lba: float = 0.10,  # Cross-Modal lexical alignment weight
        lba_max_pairs: Optional[
            int
        ] = 20000,  # Avoid O(N^2) complexity for extremely long sequences
        predictor_type: str = "t5",  # "micro" | "sliding" | "t5"
        predictor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.context = context_encoder
        self.target = target_encoder
        self.target.load_state_dict(self.context.state_dict(), strict=True)
        for p in self.target.parameters():
            p.requires_grad = False

        predictor_kwargs = predictor_kwargs or {}

        if predictor_type.lower() == "t5":
            model_pred = T5ForConditionalGeneration.from_pretrained("t5-base")
            t5_encoder_pred = model_pred.encoder
            print("Using predictor: T5")
            self.predictor = PredictorT5Full(
                t5_encoder=t5_encoder_pred,
                d_model=d_model,
                dropout=predictor_kwargs.get("dropout", 0.1),
            )
        elif predictor_type.lower() == "sliding":
            self.predictor = PredictorSlidingWindowAttn(
                d_model=d_model,
                nhead=predictor_kwargs.get("nhead", 4),
                window_overlap=predictor_kwargs.get("window", 16),
                dropout=predictor_kwargs.get("dropout", 0.2),
            )
            print("Using predictor: Sliding Window Attn")
        else:
            self.predictor = PredictorMicroAttn(
                d_model=d_model,
                nhead=predictor_kwargs.get("nhead", 4),
                window=predictor_kwargs.get("window", None),
                min_keys=predictor_kwargs.get("min_keys", 24),
                base_window=predictor_kwargs.get("base_window", 8),
                dropout=predictor_kwargs.get("dropout", 0.2),
            )
            print("Using predictor: micro-attn")

        # Lexical Bridge Alignment (LBA) configuration
        self.lambda_lba = float(lambda_lba)
        self.lba_max_pairs = lba_max_pairs

        # Switches for structural hint injection into h0
        self.use_symbolic_hint2h0 = bool(
            predictor_kwargs.get("use_symbolic_hint2h0", True)
        )
        self.use_micro_rel_window = bool(
            predictor_kwargs.get("micro_rel_window", False)
        )
        self.hint2h0_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _is_prefix_path(a: List[str], b: List[str]) -> bool:
        if len(a) > len(b):
            return False
        return a == b[: len(a)]

    @staticmethod
    def _lexical_contains(a: str, b: str) -> bool:
        if not a or not b:
            return False
        return (a in b) or (b in a)

    def _build_relational_visible_indices(
        self,
        item: Dict,
        idx_masked: torch.Tensor,
        device: torch.device,
    ) -> List[torch.Tensor]:
        tokens: List[Tuple[str, int]] = item["tokens"]
        seg_meta: List[Tuple[int, int]] = item["seg_meta"]

        # Prefer value_paths for lexicalized tree relations, fallback to type_paths.
        path_src = item.get("value_paths", []) or item.get("type_paths", [])
        gid2path: Dict[int, List[str]] = {
            int(p["current_id"]): [str(x).lower() for x in p.get("paths", [])]
            for p in path_src
            if "current_id" in p
        }

        out: List[torch.Tensor] = []
        for q in idx_masked.tolist():
            tok_q, gid_q = tokens[q]
            seg_q, mod_q = seg_meta[q]
            tok_q_norm = str(tok_q).lower()
            path_q = gid2path.get(int(gid_q), [])
            parent_q = path_q[:-1] if len(path_q) >= 2 else []

            picked = {q}

            for j, (tok_j, gid_j) in enumerate(tokens):
                if j == q:
                    continue

                seg_j, mod_j = seg_meta[j]
                tok_j_norm = str(tok_j).lower()

                # Cross-modality lexical bridge: water <-> travel_water (both directions)
                if mod_j != mod_q and self._lexical_contains(tok_q_norm, tok_j_norm):
                    picked.add(j)
                    continue

                # Same side tree relations: ancestors/descendants/siblings in one segment tree.
                if mod_j == mod_q and seg_j == seg_q:
                    path_j = gid2path.get(int(gid_j), [])
                    parent_j = path_j[:-1] if len(path_j) >= 2 else []
                    if path_q and path_j:
                        if (
                            self._is_prefix_path(path_q, path_j)
                            or self._is_prefix_path(path_j, path_q)
                            or (parent_q and parent_q == parent_j)
                        ):
                            picked.add(j)

            idx_tensor = torch.tensor(sorted(picked), device=device, dtype=torch.long)
            out.append(idx_tensor)

        return out

    @torch.no_grad()
    def update_target(self, m: float = 0.996):
        """
        Convenience method to apply EMA update to the target encoder.
        """
        ema_update(self.target, self.context, m=m)

    def _cosine_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine distance loss between two sets of embeddings.
        """
        a = F.normalize(a, dim=-1, eps=1e-8)
        b = F.normalize(b, dim=-1, eps=1e-8)
        return 1.0 - (a * b).sum(dim=-1).mean()

    # --------- Forward (batch-by-default) ---------
    def forward(
        self,
        items_full: Union[Dict, List[Dict]],
        mask_flags: Union[List[int], List[List[int]]],
        ema_m: float = 0.996,
    ):
        """
        Perform a forward pass through the JEPA model.

        Args:
            items_full:
                - Dict (single sample) or List[Dict] (batched samples).
            mask_flags:
                - List[int] (single sample) or List[List[int]] (batched samples).
            ema_m: Exponential Moving Average momentum for target network update.

        Returns:
            A dictionary containing:
            {"loss": <total_loss>, "num_masked": <total_masked_tokens>, "loss_main": <mse_loss_only>}
        """
        device = next(self.parameters()).device

        # Normalize inputs to batch format
        if isinstance(items_full, dict):
            items_full = [items_full]
            assert isinstance(mask_flags, list) and (
                len(mask_flags) == len(items_full[0]["tokens"])
            ), "mask_flags must be a list[int] for single sample"
            mask_flags_batch = [mask_flags]  # type: ignore
        else:
            mask_flags_batch = mask_flags  # type: ignore
            assert isinstance(mask_flags_batch, list) and len(mask_flags_batch) == len(
                items_full
            ), "mask_flags_batch length mismatch"

        B = len(items_full)

        # 1) Context (masked): Encode the entire batch at once (fused in StructuralTransformer)
        items_ctx = []
        for it, mf in zip(items_full, mask_flags_batch):
            x = dict(it)
            x["mask_flags"] = mf
            items_ctx.append(x)

        if hasattr(self.context.sgat, "record_attn"):
            self.context.sgat.record_attn = False
        out_ctxB = self.context.forward_batch(
            items_ctx
        )  # {'h_fused':[B,Lmax,D], 'lengths':[B], 'outs':[...]}
        H_ctxB, Ls = out_ctxB["h_fused"], out_ctxB["lengths"].tolist()

        # 2) Target (full sequence, stop-gradient)
        if hasattr(self.target.sgat, "record_attn"):
            self.target.sgat.record_attn = False
        with torch.no_grad():
            out_tgtB = self.target.forward_batch(items_full)
        H_tgtB = out_tgtB["h_fused"].detach()

        # 3) For each sample: select valid mask indices, run predictor, and accumulate loss
        zpred_list, ztgt_list = [], []
        lba_losses = []

        for b in range(B):
            L = Ls[b]
            if L == 0:
                continue

            h_ctx = H_ctxB[b, :L, :]
            h_tgt = H_tgtB[b, :L, :]

            # Semantic mask & masked positions
            sem = out_ctxB["outs"][b].get("semantic_mask", None)
            sem = (
                sem.to(device).bool()
                if sem is not None
                else torch.ones(L, dtype=torch.bool, device=device)
            )
            kpm = ~sem  # True = masked
            mf = torch.tensor(mask_flags_batch[b], device=device, dtype=torch.bool)
            idx = torch.nonzero(sem & mf, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            force_visible_indices = None
            if self.use_micro_rel_window and isinstance(
                self.predictor, PredictorMicroAttn
            ):
                force_visible_indices = self._build_relational_visible_indices(
                    item=items_full[b],
                    idx_masked=idx,
                    device=device,
                )

            # --- INJECTION: Add symbolic hints to h_ctx at masked positions (Out-of-place) ---
            if idx.numel() > 0 and self.use_symbolic_hint2h0:
                meta_ctx = out_ctxB["outs"][b]["meta"]

                # Fetch all symbolic embeddings from SANE: pos, depth, type, segment, modality.
                inj_parts = [
                    self.context.sane.pos_emb(meta_ctx["positions"][:L]),
                    self.context.sane.depth_emb(meta_ctx["depths"][:L]),
                    self.context.sane.node_type_emb(meta_ctx["node_type_ids"][:L]),
                    self.context.sane.seg_emb(meta_ctx["seg_ids"][:L]),
                    self.context.sane.modality_emb(meta_ctx["modality_ids"][:L]),
                ]

                inj = sum(inj_parts)
                inj = self.hint2h0_ln(inj)  # [L, D]

                # Add hints to masked positions (out-of-place)
                inj_sel = inj.index_select(0, idx)  # [M, D]
                delta = torch.zeros_like(h_ctx).index_add(0, idx, inj_sel)
                h_ctx = h_ctx.clone() + delta  # NOT in-place to preserve gradient graph

            predictor_kwargs = {
                "h_ctx_full": h_ctx,
                "idx_masked": idx,
                "key_padding_mask": kpm,
            }
            if isinstance(self.predictor, PredictorMicroAttn):
                predictor_kwargs["force_visible_indices"] = force_visible_indices

            z_pred = self.predictor(**predictor_kwargs)

            z_tgt = h_tgt.index_select(0, idx)

            zpred_list.append(z_pred)
            ztgt_list.append(z_tgt)

            # Calculate Lexical Bridge Alignment (LBA) loss within each sample
            if self.lambda_lba > 0:
                seg_meta = items_full[b]["seg_meta"]
                toks = [t for t, _ in items_full[b]["tokens"]]
                valid = [
                    (i, t)
                    for i, t in enumerate(toks)
                    if t not in {"(", ")"} and not mask_flags_batch[b][i]
                ]
                if valid:
                    mod = torch.tensor([m for _, m in seg_meta], device=device)
                    table = {}
                    for i, t in valid:
                        key = t.lower()
                        table.setdefault((key, int(mod[i].item())), []).append(i)
                    pairs: List[Tuple[int, int]] = []
                    for (key, m0), idxs0 in table.items():
                        m1 = 1 - m0
                        if (key, m1) in table:
                            for i0 in idxs0:
                                for j0 in table[(key, m1)]:
                                    pairs.append((i0, j0))
                    if pairs:
                        if (
                            self.lba_max_pairs is not None
                            and len(pairs) > self.lba_max_pairs
                        ):
                            perm = torch.randperm(len(pairs), device=device)[
                                : self.lba_max_pairs
                            ]
                            pairs = [pairs[k.item()] for k in perm]
                        i_idx = torch.tensor([i for i, _ in pairs], device=device)
                        j_idx = torch.tensor([j for _, j in pairs], device=device)
                        lba = self._cosine_loss(
                            h_ctx.index_select(0, i_idx), h_ctx.index_select(0, j_idx)
                        )
                        lba_losses.append(lba)

        # If there are no valid masked positions, return zero loss
        if not zpred_list:
            return {
                "loss": H_ctxB.new_zeros(()),
                "num_masked": torch.tensor(0, device=device),
            }

        z_pred = torch.cat(zpred_list, dim=0)
        z_tgt = torch.cat(ztgt_list, dim=0)
        loss = F.mse_loss(z_pred, z_tgt)
        if self.lambda_lba > 0 and lba_losses:
            loss = loss + self.lambda_lba * torch.stack(lba_losses).mean()

        return {
            "loss": loss,
            "num_masked": torch.tensor(z_pred.size(0), device=device),
            "loss_main": loss.detach(),
        }
