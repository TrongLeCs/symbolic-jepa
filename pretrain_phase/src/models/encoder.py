from typing import Dict, List, Any
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    A wrapper module that connects:
      - SANE: Outputs initial embeddings `h0` [L, D], `semantic_mask` [L], and `meta` {...}.
      - StructuralTransformer (SGAT): Takes input sequences `x` [B, L, D], raw bias matrices `raw_bias` [B, L, L, C], and an optional `key_padding_mask` [B, L].
      - structural_prior: A builder function/module to create the structural prior matrices [L, L, C].
    """

    def __init__(self, sane: nn.Module, sgat: nn.Module, structural_prior: nn.Module):
        super().__init__()
        self.sane = sane
        self.sgat = sgat
        self.structural_prior = structural_prior
        self.export_bias: bool = (
            False  # Allows exporting the raw_bias tensor for analysis
        )

    # --------- Helper Methods ---------

    @torch.no_grad()
    def _build_bias_single(
        self, item: Dict, L: int, C: int, device, dtype
    ) -> torch.Tensor:
        """
        Generate the structural prior matrix [L, L, C] for a single data sample,
        and cast it to the correct device and data type.
        """
        # Directly call the provided structural prior builder
        B_init, _ = self.structural_prior(item)

        if B_init is None:
            B_init = torch.zeros((L, L, C), device=device, dtype=dtype)
        if B_init.device != device:
            B_init = B_init.to(device)
        if B_init.dtype != dtype:
            B_init = B_init.to(dtype)
        return B_init

    def _get_bias_channels(self) -> int:
        """
        Retrieve the number of bias channels from the SGAT configuration.
        Falls back to 3 if not found.
        """
        # Try to read the number of channels from the config, fallback to 3
        if hasattr(self.sgat, "bias_channels"):
            try:
                return int(self.sgat.bias_channels)
            except Exception:
                pass
        if hasattr(self.sgat, "cfg") and hasattr(self.sgat.cfg, "bias_channels"):
            return int(self.sgat.cfg.bias_channels)
        return 3

    # --------- Batched Forward ---------

    def forward_batch(self, items: List[Dict]) -> Dict[str, Any]:
        """
        Process multiple samples in a batch and run them through the StructuralTransformer.

        Returns a dictionary containing:
          - h_fused: [B, L_max, D] (Padded fused hidden states)
          - lengths: [B]           (Actual sequence length of each sample before padding)
          - outs:    List[Dict]    (Encoder outputs for each sample: {'h0', 'semantic_mask', 'meta'})
        """
        assert len(items) > 0, "forward_batch expects a non-empty list of items"
        B = len(items)

        # 1) Encode each sample using SANE (each sample may have a different length L_b)
        outs0: List[Dict[str, torch.Tensor]] = [self.sane(it) for it in items]
        Ls = [o["h0"].size(0) for o in outs0]
        L_max = max(Ls) if Ls else 1

        device = outs0[0]["h0"].device
        dtype = outs0[0]["h0"].dtype
        D = outs0[0]["h0"].size(-1)
        C = self._get_bias_channels()

        # 2) Initialize batched tensors (buffers)
        X = torch.zeros((B, L_max, D), device=device, dtype=dtype)
        R = torch.zeros((B, L_max, L_max, C), device=device, dtype=dtype)
        KPM = torch.ones(
            (B, L_max), device=device, dtype=torch.bool
        )  # True = mask/pad (default is entirely padded)

        # 3) Populate the batch tensors with individual sample data
        for b, (it, o) in enumerate(zip(items, outs0)):
            L = Ls[b]
            if L == 0:
                continue

            # Hidden states (h0)
            X[b, :L] = o["h0"]

            # semantic_mask: 1=keep, 0=mask => key_padding_mask is True at masked positions
            sem = o.get("semantic_mask", None)
            if sem is not None:
                KPM[b, :L] = sem.to(device) == 0
            else:
                KPM[b, :L] = (
                    False  # If no semantic_mask is provided, keep all actual tokens (no padding)
                )

            # Insert the structural bias [L, L, C] into the top-left corner of the padded matrix
            B_init = self._build_bias_single(it, L, C, device, dtype)  # [L, L, C]
            R[b, :L, :L, :] = B_init

        # 4) Forward pass through the StructuralTransformer (SGAT)
        H = self.sgat(X, R, key_padding_mask=KPM)  # [B, L_max, D]

        out = {
            "h_fused": H,
            "lengths": torch.as_tensor(Ls, device=device),
            "outs": outs0,
        }
        # Used for logging the "contribution amplitude" of the biases
        if getattr(self, "export_bias", False):
            out["raw_bias"] = R
        return out
