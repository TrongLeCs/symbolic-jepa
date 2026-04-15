# =============================
# src/train.py (updated)
# =============================
from __future__ import annotations
import os

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.data import (
    load_ast_npz,
    load_dfg_npz,
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
        wandb.init(project="Decoder-logic-jepa", name=cfg.wandb_run_name)

    # 1) Load raw data
    train_ex, eval_ex = load_raw_examples(paths.train_json, paths.eval_json)
    print(f"Train samples: {len(train_ex)}. Eval samples: {len(eval_ex)}.")

    # 2) Tokenizer & model
    tokenizer = load_tokenizer(cfg.model_name)

    if struct.enable:
        print("Using T5WithStructHeads model with structural supervision.")
        model = T5WithStructHeads.from_t5_pretrained(
            cfg.model_name,  # e.g., "t5-base"
            enable_dfg=bool(getattr(struct, "enable_dfg", True)),
            enable_ast=bool(getattr(struct, "enable_ast", True)),
            alpha_dfg=struct.alpha_dfg,
            alpha_ast=struct.alpha_ast,
            num_node_types=struct.num_node_types,
            max_ast_depth=struct.max_ast_depth,
            dfg_bits=struct.dfg_bits,
            ast_path_bits=struct.ast_path_bits,
            wandb_run_name = cfg.wandb_run_name
        )

        if not bool(getattr(struct, "enable_dfg", True)):
            model.alpha_dfg = 0.0
            for p in [
                *model.proj_dfg.parameters(),
                *model.dfg_weight1.parameters(),
                *model.dfg_weight2.parameters(),
                *model.dfg_b1.parameters(),
                *model.dfg_b2.parameters(),
            ]:
                p.requires_grad = False
            model.dfg_b3.requires_grad = False

        if not bool(getattr(struct, "enable_ast", True)):
            model.alpha_ast = 0.0
            for p in [*model.proj_ast.parameters(), *model.ast_path_head.parameters()]:
                p.requires_grad = False
    else:
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    model.to(cfg.device)

    # 3) Datasets + preprocessing (per split supervision)
    train_ds, eval_ds = make_datasets(train_ex, eval_ex)

    ast_train = load_ast_npz(getattr(paths, "ast_npz_train", None))
    ast_eval = load_ast_npz(getattr(paths, "ast_npz_eval", None))
    dfg_train = load_dfg_npz(getattr(paths, "dfg_npz_train", None))
    dfg_eval = load_dfg_npz(getattr(paths, "dfg_npz_eval", None))

    preprocess_train = make_preprocess_func(
        tokenizer,
        source_max_length=cfg.source_max_len,
        target_max_length=cfg.target_max_len,
        ast_map=ast_train,
        dfg_map=dfg_train,
    )
    preprocess_eval = make_preprocess_func(
        tokenizer,
        source_max_length=cfg.source_max_len,
        target_max_length=cfg.target_max_len,
        ast_map=ast_eval,
        dfg_map=dfg_eval,
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
