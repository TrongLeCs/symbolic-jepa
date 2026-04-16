from typing import Dict, List, Tuple
import random


def build_blocks_simple(item_mix: Dict) -> List[List[int]]:
    """
    Build and return a list of blocks, where each block contains global token indices.
    In this simple implementation: 1 block = 1 meaningful token (excluding parentheses/brackets).
    """
    tokens: List[Tuple[str, int]] = item_mix["tokens"]
    blocks: List[List[int]] = []
    for i, (tok, _) in enumerate(tokens):
        if tok in {"(", ")"}:
            continue
        if (
            tok == "<extra_id_0>"
        ):  # Skip if such separator is present in tokens (rarely happens)
            continue
        blocks.append([i])
    return blocks


def sample_mask_flags(
    item_mix: Dict, mask_ratio: float = 0.2, seed: int | None = None
) -> List[int]:
    """
    Randomly sample blocks of tokens to mask based on the specified mask_ratio.
    Returns a binary list (mask_flags) of length L, where 1 indicates a masked token
    and 0 indicates an unmasked token.
    """
    rng = random.Random(seed)
    blocks = build_blocks_simple(item_mix)
    n_mask = max(1, int(round(len(blocks) * mask_ratio)))
    chosen = set(
        idx for blk in rng.sample(blocks, k=min(n_mask, len(blocks))) for idx in blk
    )
    L = len(item_mix["tokens"])
    mask_flags = [1 if i in chosen else 0 for i in range(L)]
    # Brackets are never masked (flag is always 0)
    for i, (tok, _) in enumerate(item_mix["tokens"]):
        if tok in {"(", ")"}:
            mask_flags[i] = 0
    return mask_flags
