# =============================
# src/callbacks.py
# =============================
from __future__ import annotations
from transformers import TrainerCallback


class CustomProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Loss: {logs['loss']:.4f}", end="")


class PrintEvalDatasetCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        print("Evaluating dataset:")
        for batch in kwargs["eval_dataloader"]:
            print(batch)
            break


class ContiguousCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, param in model.named_parameters():
            if param.requires_grad and not param.is_contiguous():
                param.data = param.data.contiguous()