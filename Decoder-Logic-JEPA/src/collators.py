# =============================
# src/collators.py
# =============================
from __future__ import annotations
from typing import Any, Dict, List
import torch
from transformers import DataCollatorForSeq2Seq
import torch.nn.functional as F


class DataCollatorForSeq2SeqStruct(DataCollatorForSeq2Seq):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        B = len(features)

        # 1) Remove AST/DFG to prevent super() from interfering
        ast_raw = [f.pop("ast_paths", None) for f in features]
        dfg_raw = [f.pop("dfg_links", None) for f in features]

        # 2) Call parent class to pad input/labels (labels padding -> -100)
        batch = super().__call__(features)

        labels = (
            batch["labels"]
            if torch.is_tensor(batch["labels"])
            else torch.as_tensor(batch["labels"], dtype=torch.long)
        )
        batch["labels"] = labels

        # 3) Infer Lm1_max from labels (uniformly for the whole batch)
        L_out_max = labels.size(1)  # (B, L_out_max)
        Lm1_max = max(L_out_max - 1, 0)

        # 4) AST: pad/crop -> (B, Lm1_max, D) with -1
        if any(ap is not None for ap in ast_raw):
            # Prioritize getting D from model; otherwise, infer from data
            D = getattr(self.model, "max_ast_depth", None)
            if D is None:
                D = 0
                for ap in ast_raw:
                    if ap and len(ap) > 0 and isinstance(ap[0], (list, tuple)):
                        D = max(D, len(ap[0]))
            if D == 0:
                # avoid creating tensor (B, Lm1_max, 0) if model expects D>0
                D = 1

            ast_batch = torch.full((B, Lm1_max, D), -1, dtype=torch.long)

            for i, ap in enumerate(ast_raw):
                if not ap:
                    continue
                arr = torch.as_tensor(
                    ap, dtype=torch.long
                )  # (Lm1_i, D_i?) hoặc (Lm1_i,)
                if arr.dim() == 1:
                    arr = arr.unsqueeze(-1)  # (Lm1_i, 1)
                if arr.dim() != 2:
                    raise ValueError(f"ast_paths[{i}] must be 2D, got {arr.shape}")
                Lm1_i, D_i = arr.shape

                # Normalize D
                if D_i != D:
                    if D_i > D:
                        arr = arr[:, :D]
                    else:
                        arr = F.pad(arr, (0, D - D_i), value=-1)

                # Crop/pad time dimension
                l = min(Lm1_i, Lm1_max)
                if l > 0:
                    ast_batch[i, :l, :] = arr[:l, :]
            batch["ast_paths"] = ast_batch  # (B, Lm1_max, D)

        # 5) DFG: pad/crop -> (B, Lm1_max, Lm1_max) with -1
        if any(dm is not None for dm in dfg_raw):
            dfg_batch = torch.full((B, Lm1_max, Lm1_max), -1, dtype=torch.float)
            for i, dm in enumerate(dfg_raw):
                if not dm:
                    continue
                M = torch.as_tensor(dm, dtype=torch.float)  # (Lm1_i, Lm1_i)
                if M.dim() != 2 or M.size(0) != M.size(1):
                    raise ValueError(f"dfg_links[{i}] must be square, got {M.shape}")
                Lm1_i = M.size(0)
                l = min(Lm1_i, Lm1_max)
                if l > 0:
                    dfg_batch[i, :l, :l] = M[:l, :l]
            batch["dfg_links"] = dfg_batch  # (B, Lm1_max, Lm1_max)

        return batch
