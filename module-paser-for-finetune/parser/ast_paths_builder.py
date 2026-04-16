from __future__ import annotations
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from transformers import T5TokenizerFast

# AST builder that aligns FOL tokens -> T5 subword pieces (1:1 with decoder states)
from parser.ast_paths_api import (
    DEFAULT_TYPE_VOCAB,
    build_ast_paths_tokenlevel,
)


class ASTPathsDatasetBuilderASTOnly:
    """
    Build a dataset where AST Path supervision is at the T5 *piece level*,
    aligned 1:1 with decoder hidden states (no node pooling). Only AST Path
    is produced; DFG is intentionally omitted.

    Input JSON must be a list of records with fields:
      - topic: int
      - nl:    str (optional, kept for reference)
      - fol:   str (required)

    For each record:
      1) Parse FOL and compute AST paths at piece-level via build_ast_paths_tokenlevel.
      2) Create labels (decoder targets) from the tokenizer pieces; optionally append EOS.
      3) Convert piece-level AST paths to Lm1 length used in teacher forcing
         (Lm1 = L_out - 1, with L_out = len(labels)).

    Saved .npz as object arrays to accommodate variable lengths:
      - topic_ids: (N,) int64
      - nl_texts:  (N,) object[str]
      - fol_texts: (N,) object[str]
      - fol_tokens: (N,) object[list[str]]          # FOL tokens (not subwords)
      - labels: (N,) object[np.ndarray (L_out,)]    # piece-level + optional EOS
      - ast_paths: (N,) object[np.ndarray (Lm1, D)] # piece-level AST paths after Lm1 cut
      - piece_ids: (N,) object[list[int]]           # tokenizer piece ids (no EOS)
      - piece2tok: (N,) object[list[int]]           # mapping each piece -> FOL token idx
      - offsets:  (N,) object[list[Tuple[int,int]]] # offsets in text_norm

    Meta fields:
      - max_depth, causal, type_vocab (keys/vals), tokenizer_dir, add_eos
    """

    def __init__(
        self,
        tokenizer,
        max_depth: int = 8,
        max_length: int = 256,
        type_vocab: Optional[Dict[str, int]] = None,
    ):
        self.max_depth = int(max_depth)
        self.max_length = int(max_length)
        self.type_vocab = (
            dict(type_vocab) if type_vocab is not None else DEFAULT_TYPE_VOCAB
        )
        self.tokenizer = tokenizer

        if getattr(self.tokenizer, "eos_token_id", None) is None:
            # Ensure EOS exists; T5 typically uses </s>
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})

    # ---------- core per-record ----------
    def _process_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        if "topic" not in rec or "fol" not in rec:
            raise ValueError("Record thiếu 'topic' hoặc 'fol'")

        topic_id = int(rec["topic"])  # LongTensor (bsz,)
        nl_text = str(rec.get("nl", ""))
        fol = str(rec["fol"]).strip()
        if not fol:
            raise ValueError(f"FOL rỗng cho topic={topic_id}")

        # 1) Build piece-level AST paths & alignment (no specials added inside)
        out = build_ast_paths_tokenlevel(
            expr=fol,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_depth=self.max_depth,
            type_vocab=self.type_vocab,
        )
        # unpack
        text_norm: str = out["text_norm"]
        fol_tokens: List[str] = out["fol_tokens"]
        input_ids_piece: List[int] = out["input_ids"]  # no EOS
        piece_tokens: List[str] = self.tokenizer.convert_ids_to_tokens(input_ids_piece)
        offsets: List[Tuple[int, int]] = out["offsets"]
        piece2tok: List[int] = out["piece2fol"]
        ast_paths_piece: np.ndarray = out["ast_paths_t5"]  # shape (1, L_piece, D)
        L_piece = len(input_ids_piece)

        # 2) Labels (decoder targets): no EOS appended
        labels = np.asarray(input_ids_piece, dtype=np.int64)
        L_out = int(labels.shape[0])
        Lm1 = L_out - 1

        # 3) Cut AST to Lm1 (teacher forcing length)
        #    Lm1 == L_piece - 1, drop the last row
        ast_piece_lm1: np.ndarray
        if ast_paths_piece.shape[1] != (Lm1 + 1):
            raise RuntimeError(
                f"Shape mismatch: ast_paths L={ast_paths_piece.shape[1]} vs Lm1+1={Lm1+1} (expected L=Lm1+1)"
            )
        ast_piece_lm1 = ast_paths_piece[0, :Lm1]

        return {
            "topic_id": topic_id,
            "nl_text": nl_text,
            "fol_text": fol,
            "fol_tokens": fol_tokens,
            "labels": labels,  # (L_out,)
            "ast_paths": ast_piece_lm1.astype(np.int64, copy=False),  # (Lm1, D)
            "piece_ids": input_ids_piece,  # no EOS
            "piece_tokens": piece_tokens,
            "piece2tok": piece2tok,
            "offsets": offsets,
        }

    # ---------- main build/save ----------
    def build_and_save(
        self,
        input_json_path: str | Path,
        output_npz_path: str | Path,
        error_log_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        input_json_path = Path(input_json_path)
        output_npz_path = Path(output_npz_path)

        data = json.loads(input_json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("File JSON gốc phải là một list các record.")

        topic_ids: List[int] = []
        nl_texts: List[str] = []
        fol_texts: List[str] = []
        fol_tokens_list: List[List[str]] = []
        labels_list: List[np.ndarray] = []
        ast_list: List[np.ndarray] = []
        piece_ids_list: List[List[int]] = []
        piece2tok_list: List[List[int]] = []
        offsets_list: List[List[Tuple[int, int]]] = []

        errors: List[Dict[str, Any]] = []
        output_txt_path = output_npz_path.with_suffix(".txt")
        with output_txt_path.open("w", encoding="utf-8") as txt_f:
            for idx, rec in enumerate(data):
                try:
                    obj = self._process_one(rec)
                    topic_ids.append(obj["topic_id"])
                    nl_texts.append(obj["nl_text"])
                    fol_texts.append(obj["fol_text"])
                    fol_tokens_list.append(obj["fol_tokens"])
                    labels_list.append(obj["labels"])
                    ast_list.append(obj["ast_paths"])
                    piece_ids_list.append(obj["piece_ids"])
                    piece2tok_list.append(obj["piece2tok"])
                    offsets_list.append(obj["offsets"])

                    # TXT log
                    txt_f.write(f"=== Record {idx} ===\n")
                    txt_f.write(f"topic_id: {obj['topic_id']}\n")
                    txt_f.write(f"nl_text: {obj['nl_text']}\n")
                    txt_f.write(f"fol_text: {obj['fol_text']}\n")
                    txt_f.write(f"piece_ids: {obj['piece_ids']}\n")
                    txt_f.write(f"piece_tokens: {obj['piece_tokens']}\n")
                    txt_f.write(
                        f"labels (L_out={len(obj['labels'])}): {obj['labels'].tolist()}\n"
                    )
                    txt_f.write(f"ast_paths shape: {obj['ast_paths'].shape}\n")
                    txt_f.write(f"ast_paths:\n{obj['ast_paths']}\n")
                    txt_f.write(f"piece2tok: {obj['piece2tok']}\n")
                    txt_f.write(f"offsets: {obj['offsets']}\n\n")

                except Exception as e:
                    err = {
                        "topic": rec.get("topic", None),
                        "nl": rec.get("nl", None),
                        "fol": rec.get("fol", None),
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                    errors.append(err)

        # Save .npz (object arrays)
        np.savez(
            output_npz_path,
            topic_ids=np.asarray(topic_ids, dtype=np.int64),
            nl_texts=np.asarray(nl_texts, dtype=object),
            fol_texts=np.asarray(fol_texts, dtype=object),
            fol_tokens=np.asarray(fol_tokens_list, dtype=object),
            labels=np.asarray(labels_list, dtype=object),
            ast_paths=np.asarray(ast_list, dtype=object),
            # meta
            max_depth=np.int64(self.max_depth),
            type_vocab_keys=np.asarray(list(self.type_vocab.keys()), dtype=object),
            type_vocab_vals=np.asarray(list(self.type_vocab.values()), dtype=np.int64),
            tokenizer_dir=np.asarray(str(self.tokenizer.name_or_path), dtype=object),
            add_eos=np.int8(0),
        )

        # Write error log if needed
        if errors and error_log_path is None:
            error_log_path = output_npz_path.with_suffix(
                output_npz_path.suffix + ".errors.jsonl"
            )
        if errors and error_log_path is not None:
            with Path(error_log_path).open("w", encoding="utf-8") as f:
                for e in errors:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

        return {
            "num_records": len(data),
            "num_ok": len(topic_ids),
            "num_errors": len(errors),
            "output_npz": str(output_npz_path),
            "error_log": str(error_log_path) if errors else None,
        }


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Build tokenizer-level AST dataset (.npz) from JSON for StructCoder (no pooling)"
    )
    parser.add_argument("--input", type=str, default="data/samples.json")
    parser.add_argument("--output", type=str, default="data/ast/samples_ast_paths.npz")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="t5-base",
        help="Model name or directory of T5TokenizerFast",
    )
    parser.add_argument("--max_depth", type=int, default=10, help="Max AST depth")
    parser.add_argument("--max_length", type=int, default=256, help="Max length")
    parser.add_argument("--causal", default=False, action="store_true", help="Enable causal masking for AST")
    parser.add_argument("--no_eos", default=False, action="store_true", help="Do not append EOS to labels")
    parser.add_argument(
        "--error_log",
        default="val_errors.jsonl",
        help="Optional .errors.jsonl file path",
    )

    args = parser.parse_args()
    
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer)

    builder = ASTPathsDatasetBuilderASTOnly(
        tokenizer=tokenizer,
        max_depth=args.max_depth,
        max_length=args.max_length,
        causal=bool(args.causal),
        add_eos=not args.no_eos,
    )

    stats = builder.build_and_save(args.input, args.output, args.error_log)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
