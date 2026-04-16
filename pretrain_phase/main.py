import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import wandb
from src.engine.solver import Solver
from src.models.sane import SANEConfig
from src.models.sgat import SGATConfig


ARTIFACT_ENV = "LOGIC_JEPA_ARTIFACTS_DIR"


def _artifact_default(local_path: str, artifact_path: str) -> str:
    root = os.getenv(ARTIFACT_ENV, "").strip()
    if not root:
        return local_path
    return str(Path(root) / artifact_path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def parse_args():
    p = argparse.ArgumentParser()

    # ================= Data / IO =================
    p.add_argument("--train_path", default="./data/train_dataset.jsonl")
    p.add_argument("--val_path", default="./data/val_dataset.jsonl")
    p.add_argument("--data_dir", default="saved_data")
    p.add_argument(
        "--model_dir",
        default=_artifact_default("saved_models", "encoder/saved_models"),
    )

    # ================= Training =================
    p.add_argument("--train", type=str2bool, default=True)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--accumulation_steps", type=int, default=12)
    p.add_argument("--num_epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ema_decay", type=float, default=0.996)  # used for later stages
    p.add_argument("--ema_decay_warm", type=float, default=0.9995)  # used during warm-up
    p.add_argument("--t5_ft_start_epoch", type=int, default=3)  # unfreeze T5 FT from epoch 3
    p.add_argument("--log_every", type=int, default=20)

    # ================= Predictor Configuration =================
    p.add_argument(
        "--predictor_type",
        type=str,
        default="t5",
        choices=["t5", "micro", "sliding"],
        help="Select Predictor architecture",
    )

    # -----------------------------------------------------------
    # [COMMON: t5, micro, sliding]
    # -----------------------------------------------------------
    p.add_argument(
        "--predictor_dropout",
        type=float,
        default=0.2,
        help="[t5, micro, sliding] Dropout rate for Predictor network. (Recommended: t5=0.1, micro/sliding=0.2)",
    )

    # -----------------------------------------------------------
    # [COMMON: micro, sliding]
    # -----------------------------------------------------------
    p.add_argument(
        "--predictor_nhead",
        type=int,
        default=4,
        help="[micro, sliding] Number of Attention Heads. (Must be a divisor of d_model, e.g., 768 is divisible by 4)",
    )

    # -----------------------------------------------------------
    # [VERSATILE: micro, sliding]
    # -----------------------------------------------------------
    p.add_argument(
        "--predictor_window",
        type=int,
        default=16,
        help="[micro]: Max window radius | [sliding]: Window overlap length",
    )

    # -----------------------------------------------------------
    # [SPECIFIC PARAMETERS: micro]
    # -----------------------------------------------------------
    p.add_argument(
        "--predictor_base_window",
        type=int,
        default=8,
        help="[micro] Initial window radius to start searching for valid tokens.",
    )
    p.add_argument(
        "--predictor_min_keys",
        type=int,
        default=20,
        help="[micro] Minimum number of valid tokens to reach before stopping window expansion.",
    )
    p.add_argument(
        "--predictor_micro_rel_window",
        type=str2bool,
        default=True,
        help="[micro] Enable NL<->FOL relation window: include sibling nodes and cross-modality nodes with partially matching names.",
    )

    # Optimizer & regularization
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ================= Model SANEConfig =================
    p.add_argument("--model_name", type=str, default="t5-base")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--fine_tune_t5", type=str2bool, default=True)
    p.add_argument("--max_segments", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # ---- Enable/disable each SANE channel (Ablation Study) ----
    p.add_argument("--use_compositional_path", type=str2bool, default=True)
    p.add_argument("--use_symbolic_feature", type=str2bool, default=True)

    # Masking
    p.add_argument("--mask_ratio", type=float, default=0.2)

    # DataLoader performance
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--persistent_workers", type=str2bool, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # ---- Hint injection into h_ctx at the Predictor ----
    p.add_argument("--use_symbolic_hint2h0", type=str2bool, default=True)

    # ================= SGAT =================
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dim_ff", type=int, default=1024)
    p.add_argument("--struct_dropout", type=float, default=0.2)
    p.add_argument("--bias_scale_init", type=float, default=1.0)
    p.add_argument("--bias_channels", type=int, default=3)

    # ---- Enable/disable SGAT channels (Ablation Study) ----
    p.add_argument("--use_csl", type=str2bool, default=True)
    p.add_argument("--use_lg", type=str2bool, default=True)
    p.add_argument("--use_nlb", type=str2bool, default=True)

    # ================= Loss weights =================
    # p.add_argument("--lambda_lba", type=float, default=0.1)
    p.add_argument("--lambda_lba", type=float, default=0.1)
    # ================= System =================
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--use_wandb", type=str2bool, default=True)

    args = p.parse_args()

    # -------- Build SANEConfig --------
    sane_cfg = SANEConfig(
        t5_name=args.model_name,
        max_seq_len=args.max_length,
        max_depth=args.max_depth,
        dropout=args.dropout,
        max_segments=args.max_segments,
        use_compositional_path=args.use_compositional_path,
        use_symbolic_feature=args.use_symbolic_feature,
    )

    # -------- Build SGATConfig --------
    sgat_cfg = SGATConfig(
        d_model=sane_cfg.d_ast,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.struct_dropout,
        bias_scale_init=args.bias_scale_init,
        bias_channels=args.bias_channels,
        use_csl=args.use_csl,
        use_lg=args.use_lg,
        use_nlb=args.use_nlb,
    )

    # -------- Paths config --------
    paths = SimpleNamespace(
        train_path=args.train_path,
        val_path=args.val_path,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )

    # -------- Training config --------
    cfg = SimpleNamespace(
        train=bool(args.train),
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        update_every_samples=args.batch_size * args.accumulation_steps,
        num_epochs=args.num_epochs,
        fine_tune_t5=args.fine_tune_t5,
        lr=args.lr,
        ema_decay=args.ema_decay,
        ema_decay_warm=args.ema_decay_warm,
        t5_ft_start_epoch=args.t5_ft_start_epoch,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        model_name=args.model_name,
        device=args.device,
        use_wandb=args.use_wandb,
        sane_cfg=sane_cfg,
        sgat_cfg=sgat_cfg,
        lambda_lba=args.lambda_lba,
        mask_ratio=args.mask_ratio,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        predictor_type=args.predictor_type,
        predictor_window=args.predictor_window,
        use_symbolic_hint2h0=args.use_symbolic_hint2h0,
        predictor_base_window=args.predictor_base_window,
        predictor_min_keys=args.predictor_min_keys,
        predictor_micro_rel_window=args.predictor_micro_rel_window,
    )

    return paths, cfg


if __name__ == "__main__":
    paths, cfg = parse_args()

    if cfg.use_wandb:
        wandb.login(key="d83175b72ab7d073e2ed4f0e60ef001c11cd4555")

    solver = Solver(paths, cfg)
    if cfg.train:
        solver.train()
