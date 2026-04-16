# =============================
# main.py
# =============================
from __future__ import annotations
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from src.train import run_training

DEFAULT_TYPE_VOCAB = {
    "FORALL": 0,
    "VAR": 1,
    "PRED": 2,
    "IMPLIES": 3,
    "EXISTS": 4,
    "AND": 5,
    "OR": 6,
    "XOR": 7,
    "IFF": 8,
    "NOT": 9,
    "GROUP": 10,
}

ARTIFACT_ENV = "LOGIC_JEPA_ARTIFACTS_DIR"


def _artifact_default(local_path: str, artifact_path: str) -> str:
    root = os.getenv(ARTIFACT_ENV, "").strip()
    if not root:
        return local_path
    return str(Path(root) / artifact_path)


def parse_args():
    p = argparse.ArgumentParser()

    # data / io
    p.add_argument("--train_json", default="data/train.json")
    p.add_argument("--eval_json", default="data/val.json")
    p.add_argument(
        "--pretrained_encoder_path",
        default=_artifact_default(
            "pretrain_model/saved_models/t5_target_encoder.pth",
            "encoder/saved_models/t5_target_encoder.pth",
        ),
    )
    p.add_argument("--cpp_npz_train", default="data/sample_cpp_paths.npz")
    p.add_argument("--cpp_npz_eval", default="data/sample_cpp_paths.npz")
    p.add_argument("--ldp_npz_train", default="data/sample_ldp_links.npz")
    p.add_argument("--ldp_npz_eval", default="data/sample_ldp_links.npz")
    p.add_argument(
        "--output_dir",
        default=_artifact_default("finetune_model", "decoder/finetune_model"),
    )

    # logging
    p.add_argument("--wandb_run_name", default="Logic-JEPA-0310")

    # training cfg
    p.add_argument("--model_name", default="t5-base")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--train_batch_size", type=int, default=18)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--use_cpu", type=int, default=False)
    p.add_argument("--use_mps_device", type=int, default=False)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--src_max_len", type=int, default=512)
    p.add_argument("--tgt_max_len", type=int, default=512)
    p.add_argument("--early_stopping_patience", type=int, default=4)

    # struct supervision
    p.add_argument("--enable_struct", type=int, default=1)
    p.add_argument("--enable_ldp", type=int, default=1)
    p.add_argument("--enable_cpp", type=int, default=1)
    p.add_argument("--num_node_types", type=int, default=len(DEFAULT_TYPE_VOCAB))
    p.add_argument("--max_cpp_depth", type=int, default=10)
    p.add_argument("--ldp_bits", type=int, default=16)
    p.add_argument("--cpp_path_bits", type=int, default=128)
    p.add_argument("--alpha_ldp", type=float, default=0.05) # 0.05
    p.add_argument("--alpha_cpp", type=float, default=0.1) # 0.1
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])

    args = p.parse_args()

    # Create attribute-style "configs" using SimpleNamespace
    paths = SimpleNamespace(
        train_json=args.train_json,
        eval_json=args.eval_json,
        pretrained_encoder_path=args.pretrained_encoder_path,
        cpp_npz_train=args.cpp_npz_train,
        cpp_npz_eval=args.cpp_npz_eval,
        ldp_npz_train=args.ldp_npz_train,
        ldp_npz_eval=args.ldp_npz_eval,
        output_dir=args.output_dir,
    )

    cfg = SimpleNamespace(
        model_name=args.model_name,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        train_bs=args.train_batch_size,
        eval_bs=args.eval_batch_size,
        source_max_len=args.src_max_len,
        target_max_len=args.tgt_max_len,
        wandb_run_name=args.wandb_run_name,
        device=args.device,
        accumulation_steps=args.accumulation_steps,
        use_mps_device=args.use_mps_device,
        use_cpu=args.use_cpu,
        early_stopping_patience=args.early_stopping_patience
    )

    struct = SimpleNamespace(
        enable=bool(args.enable_struct),
        enable_ldp=bool(args.enable_ldp),
        enable_cpp=bool(args.enable_cpp),
        num_node_types=args.num_node_types,
        max_cpp_depth=args.max_cpp_depth,
        ldp_bits=args.ldp_bits,
        cpp_path_bits=args.cpp_path_bits,
        alpha_ldp=args.alpha_ldp,
        alpha_cpp=args.alpha_cpp,
    )

    return paths, cfg, struct


if __name__ == "__main__":
    paths, cfg, struct = parse_args()
    run_training(paths, cfg, struct)
