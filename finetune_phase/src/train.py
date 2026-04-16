# =============================
# src/train.py (updated)
# =============================
from __future__ import annotations
import os

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.data import (
    load_cpp_npz,
    load_ldp_npz,
    load_raw_examples,
    make_datasets,
    make_preprocess_func,
)
from src.model import load_tokenizer, load_pretrained_encoder_weights, freeze_encoder
from src.struct_model import T5WithStructHeads
from src.collators import DataCollatorForSeq2SeqStruct
from src.callbacks import CustomProgressCallback, ContiguousCallback
import wandb


def run_training(paths, cfg, struct) -> None:

    if cfg.wandb_run_name:
        wandb.init(project="finetune-phase", name=cfg.wandb_run_name)

    # 1) Load raw data
    train_ex, eval_ex = load_raw_examples(paths.train_json, paths.eval_json)
    print(f"Train samples: {len(train_ex)}. Eval samples: {len(eval_ex)}.")

    # 2) Tokenizer & model
    tokenizer = load_tokenizer(cfg.model_name)

    if struct.enable:
        print("Using T5WithStructHeads model with structural supervision.")
        model = T5WithStructHeads.from_t5_pretrained(
            cfg.model_name,  # e.g., "t5-base"
            enable_ldp=bool(getattr(struct, "enable_ldp", True)),
            enable_cpp=bool(getattr(struct, "enable_cpp", True)),
            alpha_ldp=struct.alpha_ldp,
            alpha_cpp=struct.alpha_cpp,
            num_node_types=struct.num_node_types,
            max_cpp_depth=struct.max_cpp_depth,
            ldp_bits=struct.ldp_bits,
            cpp_path_bits=struct.cpp_path_bits,
            wandb_run_name = cfg.wandb_run_name
        )

        if not bool(getattr(struct, "enable_ldp", True)):
            model.alpha_ldp = 0.0
            for p in [
                *model.proj_ldp.parameters(),
                *model.ldp_weight1.parameters(),
                *model.ldp_weight2.parameters(),
                *model.ldp_b1.parameters(),
                *model.ldp_b2.parameters(),
            ]:
                p.requires_grad = False
            model.ldp_b3.requires_grad = False

        if not bool(getattr(struct, "enable_cpp", True)):
            model.alpha_cpp = 0.0
            for p in [*model.proj_cpp.parameters(), *model.cpp_path_head.parameters()]:
                p.requires_grad = False
    else:
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    model.to(cfg.device)

    # 3) Datasets + preprocessing (per split supervision)
    train_ds, eval_ds = make_datasets(train_ex, eval_ex)

    cpp_train = load_cpp_npz(getattr(paths, "cpp_npz_train", None))
    cpp_eval = load_cpp_npz(getattr(paths, "cpp_npz_eval", None))
    ldp_train = load_ldp_npz(getattr(paths, "ldp_npz_train", None))
    ldp_eval = load_ldp_npz(getattr(paths, "ldp_npz_eval", None))

    preprocess_train = make_preprocess_func(
        tokenizer,
        source_max_length=cfg.source_max_len,
        target_max_length=cfg.target_max_len,
        cpp_map=cpp_train,
        ldp_map=ldp_train,
    )
    preprocess_eval = make_preprocess_func(
        tokenizer,
        source_max_length=cfg.source_max_len,
        target_max_length=cfg.target_max_len,
        cpp_map=cpp_eval,
        ldp_map=ldp_eval,
    )

    train_ds = train_ds.map(preprocess_train, batched=True)
    eval_ds = eval_ds.map(preprocess_eval, batched=True)

    # 4) Optional: load/freeze encoder weights
    enc_path = getattr(paths, "pretrained_encoder_path", None)
    if enc_path and os.path.exists(enc_path):
        print("Start load pretrain.")
        missing, unexpected = load_pretrained_encoder_weights(model, enc_path)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        freeze_encoder(model)
        print("Encoder frozen (loaded from pretrained_encoder_path).")
    else:
        print(
            "[INFO] No custom encoder weights found. Using default pretrained T5 encoder (trainable)."
        )

    # 5) TrainingArguments
    args = TrainingArguments(
        output_dir=paths.output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_bs,
        gradient_accumulation_steps=cfg.accumulation_steps,
        use_cpu=bool(cfg.use_cpu),
        use_mps_device=bool(cfg.use_mps_device),
        per_device_eval_batch_size=cfg.eval_bs,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        logging_dir=os.path.join(paths.output_dir, "logs"),
        
        # needed for early stopping to work
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,
        save_safetensors=True,
        save_strategy="epoch",
        push_to_hub=False,
        report_to=["wandb"] if getattr(cfg, "wandb_run_name", None) else [],
        run_name=getattr(cfg, "wandb_run_name", None),
        optim="adamw_torch",
    )

    data_collator = DataCollatorForSeq2SeqStruct(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,  # allows dynamic padding
        return_tensors="pt",
    )

    callbacks = [
        CustomProgressCallback(),
        ContiguousCallback(),
        EarlyStoppingCallback(early_stopping_patience=int(getattr(cfg, "early_stopping_patience", 3)))
    ]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=callbacks,
    )

    trainer.train()

    save_dir = os.path.join(paths.output_dir, "T5WithStructHeads")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    print("Model saved")

    if cfg.wandb_run_name:
        wandb.finish()
