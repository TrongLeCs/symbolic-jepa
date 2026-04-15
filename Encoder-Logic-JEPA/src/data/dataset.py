from __future__ import print_function
from torch.utils.data import Dataset
from src.utils.io_utils import *
from src.data.linearizer import linearize_sample
from src.data.masking import sample_mask_flags

def _get_sample_length(sample):
    """Calculate the total number of tokens for NL and FOL to use as sorting criteria."""
    length = 0
    # Count tokens in ast_nl
    ast_nl = sample.get("ast_nl", [])
    if isinstance(ast_nl, dict):
        ast_nl = [ast_nl]
    for nl in ast_nl:
        length += len(nl.get("tokens", []))
        
    # Count tokens in ast_fol
    ast_fol = sample.get("ast_fol", [])
    if isinstance(ast_fol, dict):
        ast_fol = [ast_fol]
    for fol in ast_fol:
        length += len(fol.get("tokens", []))
        
    return length

class JepaDataset(Dataset):
    def __init__(self, paths, mode="train"):
        self.mode = mode
        if mode == "train":
            self.data = load_jsonl_dataset(paths.train_path)
        elif mode == "val":
            self.data = load_jsonl_dataset(paths.val_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Smart Batching: Sort in ascending order (shortest sentences first)
        # to avoid GPU memory shock in initial steps.
        self.data.sort(key=_get_sample_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        topic = sample.get("topic")
        ast_nl = sample.get("ast_nl", [])
        ast_fol = sample.get("ast_fol", [])

        if isinstance(ast_nl, dict):
            ast_nl = [ast_nl]

        if (
            isinstance(ast_fol, list)
            and len(ast_fol) == 1
            and isinstance(ast_fol[0], dict)
        ):
            ast_fol = ast_fol[0]

        return {
            "topic": topic,
            "ast_nl": ast_nl,
            "ast_fol": ast_fol,
        }

def jepa_collate_fn(batch, mask_ratio=0.2):
    items_full = [linearize_sample(item) for item in batch]
    mask_flags_batch = []
    
    for i, item in enumerate(items_full):
        flags = sample_mask_flags(item, mask_ratio=mask_ratio)
        mask_flags_batch.append(flags)
        
    return items_full, mask_flags_batch