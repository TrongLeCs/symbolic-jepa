import argparse
import json
import os
from pathlib import Path

import torch
from transformers import T5Tokenizer
from tqdm import tqdm

from src.struct_model import T5WithStructHeads


ARTIFACT_ENV = "LOGIC_JEPA_ARTIFACTS_DIR"


def _artifact_default(local_path: str, artifact_path: str) -> str:
    root = os.getenv(ARTIFACT_ENV, "").strip()
    if not root:
        return local_path
    return str(Path(root) / artifact_path)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Two generation presets (static parameters only)
# Note: DO NOT include pad_token_id in the preset as the tokenizer does not exist here yet.
PRESET_A = dict(
    max_length=512,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2,
    repetition_penalty=2.0,
)

PRESET_B = dict(
    do_sample=True,
    temperature=1.0,
    max_new_tokens=512,
    num_return_sequences=3,  # generate 3 samples / input
    # Other parameters if needed:
    # early_stopping=True,
    # length_penalty=1.0,
    # no_repeat_ngram_size=3,
    # repetition_penalty=1.05,
)


def decode_many(tokenizer, out_ids):
    """
    Returns a list[str] decoded from out_ids.
    Supports out_ids as a Tensor of shape [N, T] or list[list[int]].
    """
    if isinstance(out_ids, torch.Tensor):
        # out_ids: [N, T]
        seqs = out_ids.tolist()
    else:
        # list[list[int]]
        seqs = out_ids
    return [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in seqs]


# ✅ Add this function (does not affect other logic)
def normalize_keys(dataset):
    new_dataset = []
    for item in dataset:
        new_item = {}
        for k, v in item.items():
            if k == "nl":
                new_item["NL"] = v
            elif k == "fol":
                new_item["FOL"] = v
            else:
                new_item[k] = v
        new_dataset.append(new_item)
    return new_dataset


def run_inference(
    model_path: str,
    model_name: str,
    dataset_path: str,
    output_path: str,
    preset_choice: str,
):
    device = get_device()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5WithStructHeads.from_pretrained(model_path).to(device).eval()

    # Prepare dynamic preset (inject pad_token_id after the tokenizer is available)
    if preset_choice == "A":
        gen_kwargs = dict(PRESET_A)
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    elif preset_choice == "B":
        gen_kwargs = dict(PRESET_B)
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    else:
        raise ValueError("preset_choice must be one of: A, B")

    # Read data
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    result_dataset = []
    batch_size = 64  # adjust based on VRAM
    total = len(dataset)
    pbar = tqdm(total=total, desc="[INFERENCE/BATCH]", unit="sample")

    # Number of candidates / input (preset B has num_return_sequences > 1)
    k = gen_kwargs.get("num_return_sequences", 1)

    for start in range(0, total, batch_size):
        batch = dataset[start:start + batch_size]
        texts = [s["nl"] for s in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k2: v.to(device) for k2, v in enc.items()}

        with torch.no_grad():
            out_ids = model.forward2(**enc, **gen_kwargs)

        # out_ids usually have shape [batch_size * k, T]
        outputs = decode_many(tokenizer, out_ids)

        expected = len(batch) * k
        if len(outputs) != expected:
            expected = min(expected, len(outputs))

        # Assign results back to each sample
        for j, sample in enumerate(batch):
            start_idx = j * k
            end_idx = start_idx + k
            cand = outputs[start_idx:end_idx]

            if preset_choice == "A":
                sample["Predict-FOL"] = cand
                # sample["output_preset_A_top1"] = cand[0] if cand else ""
            else:
                sample["Predict-FOL"] = cand
                # sample["output_preset_B_top1"] = cand[0] if cand else ""

        result_dataset.extend(batch)
        pbar.update(len(batch))

    pbar.close()

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ✅ ONLY add this one line
    result_dataset = normalize_keys(result_dataset)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dataset, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=_artifact_default(
            "./finetune_model/T5WithStructHeads",
            "decoder/finetune_model/T5WithStructHeads",
        ),
    )
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--dataset_path", type=str, default="data/test.json")
    parser.add_argument(
        "--output_path",
        type=str,
        default=_artifact_default(
            "inference_results/test.json",
            "decoder/inference_results/test.json",
        ),
    )
    parser.add_argument(
        "--preset", type=str, choices=["A", "B"], default="B", help="Choose generation configuration: A, B"
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        preset_choice=args.preset,
    )


if __name__ == "__main__":
    main()