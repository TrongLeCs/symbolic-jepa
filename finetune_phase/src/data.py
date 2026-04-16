from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from typing import Dict, Any, List, Tuple, Optional, Callable


def load_raw_examples(
    train_json_path: str, eval_json_path: str
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Load data from JSON files
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Convert data into examples
    train_examples = [_to_example(ex) for ex in train_data]
    eval_examples = [_to_example(ex) for ex in eval_data]

    # Filter examples with valid "NL" and "FOL" data
    train_examples = [ex for ex in train_examples if ex["NL"] and ex["FOL"]]
    eval_examples = [ex for ex in eval_examples if ex["NL"] and ex["FOL"]]

    return train_examples, eval_examples


def _to_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON record into a standard example, removing fields related to CPP and LDP."""
    nl = (ex.get("nl") or "").strip()  # Get NL sentence from JSON
    fol = (ex.get("fol") or "").strip()  # Get FOL formula from JSON

    # Return standard example with "NL" and "FOL" fields
    out = {"NL": nl, "FOL": fol}

    # If there is a topic, add it to the output
    if "topic" in ex:
        out["topic"] = ex["topic"]

    # No parts related to CPP and LDP in this example
    return out


def make_datasets(train_examples, eval_examples) -> tuple[Dataset, Dataset]:
    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples)


# ----- Preprocessing Function -----
def _preprocess(
    batch: Dict[str, Any],
    *,
    tokenizer,
    source_max_length: int = 256,
    target_max_length: int = 256,
    cpp_map: Optional[Dict[int, Dict[str, Any]]] = None,
    ldp_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    nl_list = batch.get("NL", [])
    fol_list = batch.get("FOL", [])
    has_topic = "topic" in batch
    bsz = len(nl_list)

    # 1) tokenize source (NO padding)
    src = tokenizer(
        nl_list,
        max_length=source_max_length,
        truncation=True,
        padding=False,
    )

    # 2) tokenize target (NO padding) — use text_target to ensure decoder specials
    tgt = tokenizer(
        text_target=fol_list,
        max_length=target_max_length,
        truncation=True,
        padding=False,
    )

    # DO NOT set -100 here; let the collator handle it
    src["labels"] = tgt["input_ids"]  # list of lists, variable length

    # 3) attach raw supervision for CPP/LDP (NO padding/cropping)
    if cpp_map is None or ldp_map is None:
        raise ValueError("cpp_map and ldp_map are required in this pipeline.")

    cpp_paths_list, ldp_links_list = [], []
    for i in range(bsz):
        tid = int(batch["topic"][i]) if has_topic else None
        sup_cpp = cpp_map[tid]
        sup_ldp = ldp_map[tid]

        # Normalization: handle both 'cpp_paths'/'ldp_links' and legacy 'ast_paths'/'dfg_links'
        cpp_paths_key = "cpp_paths" if "cpp_paths" in sup_cpp else "ast_paths"
        ldp_links_key = "ldp_links" if "ldp_links" in sup_ldp else "dfg_links"

        cpp_paths = sup_cpp[cpp_paths_key]
        ldp_links = sup_ldp[ldp_links_key]

        # ensure it is a list of lists (variable length)
        if isinstance(cpp_paths, np.ndarray):
            cpp_paths = cpp_paths.tolist()
        if isinstance(ldp_links, np.ndarray):
            ldp_links = ldp_links.tolist()

        cpp_paths_list.append(cpp_paths)  # (Lm1_i, D_i) raw
        ldp_links_list.append(ldp_links)  # (Lm1_i, Lm1_i) raw

    src["cpp_paths"] = cpp_paths_list
    src["ldp_links"] = ldp_links_list

    return src


def make_preprocess_func(
    tokenizer,
    source_max_length: int = 256,
    target_max_length: int = 256,
    cpp_map: Optional[Dict[int, Dict[str, Any]]] = None,
    ldp_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a preprocess function (callable) for dataset.map(batched=True).
    """

    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return _preprocess(
            batch,
            tokenizer=tokenizer,
            source_max_length=source_max_length,
            target_max_length=target_max_length,
            cpp_map=cpp_map,
            ldp_map=ldp_map,
        )

    return _fn


def load_cpp_npz(npz_path: Optional[str | Path]) -> Optional[Dict[int, Dict[str, Any]]]:
    if not npz_path:
        return None
    p = Path(npz_path)
    if not p.exists():
        print(f"[WARN] CPP npz not found: {p}")
        return None

    data = np.load(p, allow_pickle=True)

    # Required fields according to the .npz file (normalize legacy names)
    topic_ids = list(map(int, data["topic_ids"].tolist()))
    labels_list = data["labels"].tolist()  # each element: np.ndarray shape (L_out,)
    
    cpp_key = "cpp_paths" if "cpp_paths" in data else "ast_paths"
    cpp_list = data[cpp_key].tolist()  # each element: np.ndarray shape (Lm1, D)

    # Meta (may or may not exist)
    max_depth = int(data.get("max_depth", np.array(10, dtype=np.int64)))

    out: Dict[int, Dict[str, Any]] = {}

    n_bad = 0
    for i, tid in enumerate(topic_ids):
        labels = labels_list[i]
        cpp_p = cpp_list[i]

        # Sanity-check: follow Approach A, cpp_paths have shape (Lm1, D) with Lm1 = L_out - 1
        L_out = int(labels.shape[0])
        Lm1_expected = L_out - 1
        Lm1_actual = int(cpp_p.shape[0])
        if Lm1_actual != Lm1_expected:
            n_bad += 1
            print(
                f"[WARN] topic_id={tid}: Lm1(cpp)={Lm1_actual} != L_out-1={Lm1_expected}"
            )

        out[tid] = {
            "labels": labels,  # (L_out,)
            "cpp_paths": cpp_p,  # (Lm1, D) — NO BOS/EOS
        }

    print(
        f"Loaded CPP supervision for {len(out)} samples (max_depth={max_depth})."
        + (f" Mismatched: {n_bad}" if n_bad else "")
    )

    return out


# ---------- Load LDP supervision from saved .npz ----------
# Compatible with LDPLinksDatasetBuilderT5 (object arrays)


def load_ldp_npz(npz_path: Optional[str | Path]) -> Optional[Dict[int, Dict[str, Any]]]:
    if not npz_path:
        return None
    p = Path(npz_path)
    if not p.exists():
        print(f"[WARN] LDP npz not found: {p}")
        return None
    data = np.load(p, allow_pickle=True)
    topic_ids = list(map(int, data["topic_ids"].tolist()))
    
    ldp_key = "ldp_links" if "ldp_links" in data else "dfg_links"
    ldp_list = data[ldp_key].tolist()  # each: (L, L) with -1 mask

    out: Dict[int, Dict[str, Any]] = {}
    for i, tid in enumerate(topic_ids):
        out[tid] = {
            "ldp_links": ldp_list[i],
        }

    print(f"Loaded LDP supervision for {len(out)} samples.")
    return out
