# =============================
# src/model.py
# =============================
from __future__ import annotations
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Tuple


def load_tokenizer(model_name: str) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return tokenizer


def load_pretrained_encoder_weights(
    model: T5ForConditionalGeneration, encoder_state_path: str
) -> tuple[list, list]:
    state = torch.load(encoder_state_path, map_location="cuda")
    missing_keys, unexpected_keys = model.encoder.load_state_dict(state, strict=False)
    return missing_keys, unexpected_keys


def freeze_encoder(model: T5ForConditionalGeneration) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False
