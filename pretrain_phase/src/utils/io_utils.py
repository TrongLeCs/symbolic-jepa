from __future__ import print_function
import random
from typing import Dict, List
import json
import os
import torch.nn as nn
import torch


def read_json(file_path):
    """
    Read and return the parsed content of a JSON file.
    Returns an empty list if the file is not found or has an invalid format.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return []


def save_json_file(data, file_path):
    """
    Save the provided data dictionary or list to a JSON file with pretty formatting.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")


def load_jsonl_dataset(jsonl_file: str) -> List[Dict]:
    """
    Read an entire .jsonl file and return a list of dictionaries.
    Skips empty lines and logs an error for any malformed JSON lines.
    """
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse error at line {line_idx}: {e}")
    print(f"Loaded {len(data)} samples from {jsonl_file}")
    return data


def batchify(data, batch_size: int):
    """
    Yield successive batches of a specified size from the given data list.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def save_jsonl_file(data, path):
    """
    Save a list of dictionaries to a JSON Lines (.jsonl) file,
    writing one JSON object per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def make_save_dir(save_dir):
    """
    Create the specified directory path if it does not already exist.
    Returns the directory path.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# ================== Utils ==================
def set_seed(seed: int = 1234):
    """
    Set the random seed for Python's random module and PyTorch
    to ensure reproducibility across runs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    """
    Count and return the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Determine and return the best available PyTorch device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
