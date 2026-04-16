from __future__ import annotations
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from typing import Tuple
import os
import torch


def load_tokenizer(
    model_name: str,
) -> Tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    return tokenizer


def load_t5_model(model_name: str) -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration.from_pretrained(model_name)


def load_pretrained_encoder_weights(
    model: T5ForConditionalGeneration, encoder_state_path: str
) -> tuple[list, list]:
    state = torch.load(encoder_state_path, map_location="cpu")
    missing_keys, unexpected_keys = model.encoder.load_state_dict(state, strict=False)
    return missing_keys, unexpected_keys


def freeze_encoder(model: T5ForConditionalGeneration) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def save_best_model(model_dict: dict, save_dir: str, tag: str):
    """
    Save the state_dict of each component in JEPA.
    tag: 'epoch3' or 'best'
    """
    os.makedirs(save_dir, exist_ok=True)
    for name, module in model_dict.items():
        if hasattr(module, "encoder"):
            state_dict = module.encoder.state_dict()
        else:
            state_dict = module.state_dict()

        save_path = os.path.join(save_dir, f"{name}_{tag}.pth")
        torch.save(state_dict, save_path)
        print(f"[Info] Saved {name} -> {save_path}")
