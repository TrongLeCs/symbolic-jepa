from copy import deepcopy
from functools import partial
from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

from src.models.sane import SANE
from src.models.sgat import SGAT, build_structural_prior
from src.models.encoder import Encoder
from src.models.jepa import LogicJEPA
from src.utils.io_utils import count_params


class ModelFactory:
    """
    A factory class responsible for instantiating and assembling the components
    of the LogicJEPA architecture.
    """

    @staticmethod
    def build_jepa(cfg, tokenizer, t5_model: T5ForConditionalGeneration) -> nn.Module:
        """
        Construct the complete LogicJEPA model including the Context Encoder,
        Target Encoder, and Predictor based on the provided configuration.
        """
        device = torch.device(cfg.device)

        # 1) SANE Encoder
        sane = SANE(cfg.sane_cfg, tokenizer=tokenizer, t5_model=t5_model).to(device)

        # 2) Structural Transformer (SGAT)
        sgat = SGAT(cfg.sgat_cfg).to(device)

        # 3) Context Encoder Stream
        encoder = Encoder(
            sane=sane,
            sgat=sgat,
            structural_prior=partial(
                build_structural_prior,
                use_csl=bool(cfg.sgat_cfg.use_csl),
                use_lg=bool(cfg.sgat_cfg.use_lg),
                use_nlb=bool(cfg.sgat_cfg.use_nlb),
            ),
        ).to(device)

        # 4) Predictor kwargs (passing structural hint flags to h0)
        predictor_kwargs: Dict[str, Any] = {
            "use_symbolic_hint2h0": bool(getattr(cfg, "use_symbolic_hint2h0", True)),
        }

        if cfg.predictor_type == "micro":
            # window < 0 indicates global attention; otherwise, use window mode
            w = getattr(cfg, "predictor_window", -1)
            w = None if (w is None or int(w) < 0) else int(w)

            base_w = int(getattr(cfg, "predictor_base_window", 8))
            # Only clamp base_window when operating in window mode
            if isinstance(w, int):
                base_w = max(1, min(base_w, w))

            predictor_kwargs.update(
                {
                    "window": w,  # Maximum search radius (None implies global)
                    "base_window": base_w,  # Initial search radius (e.g., 8)
                    "min_keys": int(
                        getattr(cfg, "predictor_min_keys", 24)
                    ),  # Minimum number of valid keys required
                    "micro_rel_window": bool(
                        getattr(cfg, "predictor_micro_rel_window", False)
                    ),
                    "nhead": int(getattr(cfg, "predictor_nhead", 4)),
                    "dropout": float(getattr(cfg, "predictor_dropout", 0.2)),
                }
            )
        elif cfg.predictor_type == "sliding":
            # Sliding Window Attention requires 'window' (for overlap size) and 'nhead'
            predictor_kwargs.update(
                {
                    "window": int(
                        getattr(cfg, "predictor_window", 16)
                    ),  # Window overlap size
                    "nhead": int(getattr(cfg, "predictor_nhead", 4)),
                    "dropout": float(getattr(cfg, "predictor_dropout", 0.2)),
                }
            )

        else:  # Default is "t5"
            predictor_kwargs.update(
                {
                    "dropout": float(getattr(cfg, "predictor_dropout", 0.1)),
                }
            )

        # 5) LogicJEPA Model Assembly
        jepa = LogicJEPA(
            context_encoder=encoder,
            target_encoder=deepcopy(encoder),
            d_model=cfg.sane_cfg.d_ast,
            lambda_lba=cfg.lambda_lba,
            predictor_type=cfg.predictor_type,
            predictor_kwargs=predictor_kwargs,
        ).to(device)

        # The target SANE encoder is always frozen (no fine-tuning)
        if hasattr(jepa.target.sane, "set_finetune"):
            jepa.target.sane.set_finetune(False)

        print(
            f"[Info] SANE channels: "
            f"comp_path={cfg.sane_cfg.use_compositional_path}, sym_feat={cfg.sane_cfg.use_symbolic_feature} | "
            f"fine_tune_t5={cfg.fine_tune_t5}"
        )
        print(f"[Info] Trainable params: {count_params(jepa):,}")
        return jepa
