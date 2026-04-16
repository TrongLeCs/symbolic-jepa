# dfg_links_packager_t5.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# === API T5-level (bạn đã có trong code trước đó) ===
# Đảm bảo module dưới đây có hàm: build_dfg_links_tokenlevel(expr, tokenizer, project_mode="first"| "all")
from parser.dfg_links_api import build_dfg_links_tokenlevel


class DFGLinksDatasetBuilderT5:
    """
    Đọc danh sách record {topic, nl, fol} và đóng gói .npz cho nhiệm vụ DFG links ở MỨC TOKENIZER (T5 pieces):
      - topic_ids: (N,) int64
      - nl_texts:  (N,) object[str]
      - text_norm: (N,) object[str]        # chuỗi normalized feed vào T5 (không specials)
      - tokens:    (N,) object[list[str]]  # T5 pieces
      - types:     (N,) object[list[str]]  # nhãn loại token (theo T5)
      - input_ids: (N,) object[np.ndarray(L_i,)]
      - offsets:   (N,) object[list[Tuple[int,int]]]  # offset mapping trong text_norm
      - piece2fol: (N,) object[np.ndarray(L_i,)]      # -1 nếu không khớp
      - dfg_links: (N,) object[np.ndarray(L_i, L_i)]  # ma trận DFG T5 (int8), -1 mask cho ignored
      - dfg_edges: (N,) object[np.ndarray(E_i,2)]     # danh sách cạnh (src,dst) với value==1
      - token_predicate_id: (N,) object[np.ndarray(L_i,)] # id theo LOCAL predicate vocab, -1 nếu không phải tên predicate
      - predicate_vocab_local_keys/vals: (N,) object[list] # vocab cục bộ (ổn định trong mẫu)

    Lưu ý:
      * Chiều L_i khác nhau giữa mẫu → dùng object arrays.
      * Không tạo vocab toàn cục ở đây; nếu cần bạn có thể hợp nhất từ các local sau khi load.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        """
        tokenizer: T5TokenizerFast (đã load)
        project_mode: 'first' | 'all' (cách chiếu edges FOL->T5 trong API; khuyên dùng 'first')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _process_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        if "topic" not in rec or "fol" not in rec:
            raise ValueError("Record thiếu 'topic' hoặc 'fol'")

        topic_id = int(rec["topic"])
        nl_text = str(rec.get("nl", ""))

        fol = str(rec["fol"]).strip()
        if not fol:
            raise ValueError(f"FOL rỗng cho topic={topic_id}")

        out = build_dfg_links_tokenlevel(
            expr=fol, tokenizer=self.tokenizer, max_length=self.max_length
        )

        # Lấy đúng các trường ở MỨC T5
        tokens_t5: List[str] = out["t5_tokens"]  # list[str]
        types_t5: List[str] = out["types_t5"]  # list[str]
        mat_t5: np.ndarray = out["dfg_links_t5"]  # (1, L, L)
        token_pred_id_t5: np.ndarray = out["token_predicate_id_t5"]  # (L,)
        pred_vocab_local: Dict[str, int] = out["predicate_vocab"]  # tên -> id (cục bộ)
        input_ids: np.ndarray = out["input_ids"]  # (L,)
        offsets: List[Tuple[int, int]] = out["offsets"]  # list[(s,e)]
        piece2fol: np.ndarray = out["piece2fol"]  # (L,)
        text_norm: str = out["text_norm"]

        # Bóc batch dim
        mat_nn = mat_t5[0]  # (L, L)

        # Trích edges (src,dst) value==1
        edges_idx = np.argwhere(mat_nn == 1).astype(np.int32)  # (E,2)

        return {
            "topic": topic_id,
            "nl": nl_text,
            "text_norm": text_norm,
            "tokens": tokens_t5,
            "types": types_t5,
            "input_ids": input_ids.astype(np.int32, copy=False),
            "offsets": offsets,
            "piece2fol": piece2fol.astype(np.int32, copy=False),
            "dfg_links": mat_nn.astype(np.int8, copy=False),
            "dfg_edges": edges_idx,  # (E,2)
            "token_pred_id_local": token_pred_id_t5.astype(np.int32, copy=False),
            "local_pred_vocab": pred_vocab_local,  # {name: lid_local}
        }

    def build_and_save(
        self,
        input_json_path: str | Path,
        output_npz_path: str | Path,
        error_log_path: Optional[str | Path] = None,
        *,
        print_matrix: bool = True,
        max_print_tokens: int = 500,
        max_print_matrix_n: int = 400,
        max_print_edges: int = 500,
        print_local_vocab: bool = True,
    ) -> Dict[str, Any]:
        import io

        input_json_path = Path(input_json_path)
        output_npz_path = Path(output_npz_path)
        output_txt_path = output_npz_path.with_suffix(".txt")

        raw = input_json_path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON input không hợp lệ: {e}")

        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                raise ValueError("File JSON phải là list các record {topic,nl,fol}.")

        processed: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        # --------- Xử lý từng record ---------
        for idx, rec in enumerate(data):
            try:
                ex = self._process_one(rec)
                processed.append(ex)
            except Exception as e:
                errors.append(
                    {
                        "index_in_file": idx,
                        "topic": rec.get("topic", None),
                        "fol": rec.get("fol", None),
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                )

        topic_ids: List[int] = []
        nl_texts: List[str] = []
        text_norms: List[str] = []
        tokens_list: List[List[str]] = []
        types_list: List[List[str]] = []
        input_ids_list: List[np.ndarray] = []
        offsets_list: List[List[Tuple[int, int]]] = []
        piece2fol_list: List[np.ndarray] = []
        dfg_list: List[np.ndarray] = []
        dfg_edges_list: List[np.ndarray] = []
        token_pred_local_list: List[np.ndarray] = []
        pred_vocab_local_keys_list: List[List[str]] = []
        pred_vocab_local_vals_list: List[List[int]] = []

        # Buffer để ghi TXT một lần
        txt_buf = io.StringIO()
        txt_buf.write(f"# DFG Links (T5 level) dump for {output_npz_path.name}\n")
        txt_buf.write(f"# Total input records: {len(data)}\n")
        if errors:
            txt_buf.write(f"# Records with errors (excluded): {len(errors)}\n")
        txt_buf.write("# --------------------------------------------\n\n")

        # --------- Thu gom & ghi TXT ---------
        for ridx, ex in enumerate(processed):
            topic_id = int(ex["topic"])
            nl_text = ex["nl"]
            text_norm = ex["text_norm"]
            tokens = ex["tokens"]
            types = ex["types"]
            mat = ex["dfg_links"]  # (L,L) int8
            edges_idx = ex["dfg_edges"]  # (E,2)
            lid_vec = ex["token_pred_id_local"]  # (L,) int32
            local_vocab: Dict[str, int] = ex["local_pred_vocab"]

            topic_ids.append(topic_id)
            nl_texts.append(nl_text)
            text_norms.append(text_norm)
            tokens_list.append(tokens)
            types_list.append(types)
            input_ids_list.append(ex["input_ids"])
            offsets_list.append(ex["offsets"])
            piece2fol_list.append(ex["piece2fol"])
            dfg_list.append(mat)
            dfg_edges_list.append(edges_idx)
            token_pred_local_list.append(lid_vec)

            inv_sorted = sorted(local_vocab.items(), key=lambda kv: kv[1])
            pred_vocab_local_keys_list.append([name for name, _ in inv_sorted])
            pred_vocab_local_vals_list.append([lid for _, lid in inv_sorted])

            # ---- Ghi từng mẫu ra TXT ----
            txt_buf.write(f"=== Record {ridx} ===\n")
            txt_buf.write(f"topic_id: {topic_id}\n")
            txt_buf.write(f"nl_text: {nl_text}\n")
            txt_buf.write(f"text_norm: {text_norm}\n")

            # tokens / types (cắt bớt nếu dài)
            if len(tokens) <= max_print_tokens:
                txt_buf.write(f"tokens ({len(tokens)}): {tokens}\n")
                txt_buf.write(f"types  ({len(types)}): {types}\n")
            else:
                txt_buf.write(
                    f"tokens ({len(tokens)}) [first {max_print_tokens}]: {tokens[:max_print_tokens]} ...\n"
                )
                txt_buf.write(
                    f"types  ({len(types)})  [first {max_print_tokens}]: {types[:max_print_tokens]} ...\n"
                )

            # token_predicate_id (LOCAL, mức T5)
            if len(lid_vec) <= max_print_tokens:
                txt_buf.write(
                    f"token_predicate_id_T5 (LOCAL, len={len(lid_vec)}): {lid_vec.tolist()}\n"
                )
            else:
                txt_buf.write(
                    f"token_predicate_id_T5 (LOCAL, len={len(lid_vec)}) [first {max_print_tokens}]: {lid_vec[:max_print_tokens].tolist()} ...\n"
                )

            # Ma trận kề
            L = mat.shape[0]
            txt_buf.write(f"dfg_links shape: {mat.shape}\n")
            PRINT_MATRIX = True
            if PRINT_MATRIX:
                write_matrix_rows(
                    mat,
                    txt_buf,
                    max_rows=50,
                    max_cols=50,
                    header="dfg_links (row-wise):",
                    pad=2,  # chỉnh cho dễ đọc
                )

            # In edges (preview)
            txt_buf.write(f"edges (value==1): {edges_idx.shape[0]} edges\n")
            show_k = min(max_print_edges, edges_idx.shape[0])
            for k in range(show_k):
                s, d = int(edges_idx[k, 0]), int(edges_idx[k, 1])
                tok_s = tokens[s] if 0 <= s < len(tokens) else "<OOB>"
                tok_d = tokens[d] if 0 <= d < len(tokens) else "<OOB>"
                txt_buf.write(f"{s} -> {d}   ({tok_s} -> {tok_d})\n")
            if edges_idx.shape[0] > show_k:
                txt_buf.write(f"... ({edges_idx.shape[0] - show_k} more edges)\n")

            # Vocab cục bộ
            if print_local_vocab:
                txt_buf.write("LOCAL PREDICATE VOCAB (id_local -> name):\n")
                for name, lid in inv_sorted:
                    txt_buf.write(f"{lid}\t{name}\n")

            txt_buf.write("\n")

        # ---- Lưu .npz ----
        np.savez(
            output_npz_path,
            topic_ids=np.asarray(topic_ids, dtype=np.int64),
            nl_texts=np.asarray(nl_texts, dtype=object),
            text_norm=np.asarray(text_norms, dtype=object),
            tokens=np.asarray(tokens_list, dtype=object),
            types=np.asarray(types_list, dtype=object),
            input_ids=np.asarray(input_ids_list, dtype=object),
            offsets=np.asarray(offsets_list, dtype=object),
            piece2fol=np.asarray(piece2fol_list, dtype=object),
            dfg_links=np.asarray(dfg_list, dtype=object),
            dfg_edges=np.asarray(dfg_edges_list, dtype=object),
            token_predicate_id=np.asarray(token_pred_local_list, dtype=object),
            predicate_vocab_local_keys=np.asarray(
                pred_vocab_local_keys_list, dtype=object
            ),
            predicate_vocab_local_vals=np.asarray(
                pred_vocab_local_vals_list, dtype=object
            ),
        )

        # ---- Ghi lỗi (TXT + JSONL) ----
        if errors:
            txt_buf.write("=== ERRORS ===\n")
            for e in errors:
                txt_buf.write(json.dumps(e, ensure_ascii=False) + "\n")
            txt_buf.write("\n")

        with output_txt_path.open("w", encoding="utf-8") as ftxt:
            ftxt.write(txt_buf.getvalue())

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
            "num_ok": len(tokens_list),
            "num_errors": len(errors),
            "output_npz": str(output_npz_path),
            "output_txt": str(output_txt_path),
            "error_log": str(error_log_path) if errors else None,
        }


def write_matrix_rows(
    mat: np.ndarray,
    buf,
    *,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    header: Optional[str] = None,
    pad: int = 2,
) -> None:
    """
    In ma trận theo từng hàng vào buffer `buf` (vd: StringIO hoặc file),
    tránh bị numpy rút gọn bằng '...'.
    """
    R, C = mat.shape
    rlim = R if max_rows is None else min(R, max_rows)
    clim = C if max_cols is None else min(C, max_cols)

    if header:
        buf.write(header + "\n")

    # In chỉ mục cột (tuỳ chọn)
    buf.write("     ")
    buf.write(" ".join(f"{j:>{pad}d}" for j in range(clim)) + "\n")

    # In từng hàng
    for i in range(rlim):
        row_str = " ".join(f"{int(mat[i, j]):>{pad}d}" for j in range(clim))
        buf.write(f"{i:>4d}: {row_str}\n")

    # Thông báo nếu cắt ngắn
    if rlim < R or clim < C:
        buf.write(f"... truncated; full size {R}x{C}\n")


def load_packed_dfg_t5(npz_path: str | Path) -> Dict[str, Any]:
    """
    Tiện ích load .npz DFG ở mức T5.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as z:
        return {
            "topic_ids": z["topic_ids"],  # (N,)
            "nl_texts": z["nl_texts"],  # (N,) object[str]
            "text_norm": z["text_norm"],  # (N,) object[str]
            "tokens": z["tokens"],  # (N,) object[list[str]]
            "types": z["types"],  # (N,) object[list[str]]
            "input_ids": z["input_ids"],  # (N,) object[np.ndarray(L_i,)]
            "offsets": z["offsets"],  # (N,) object[list[(s,e)]]
            "piece2fol": z["piece2fol"],  # (N,) object[np.ndarray(L_i,)]
            "dfg_links": z["dfg_links"],  # (N,) object[np.ndarray(L_i,L_i)]
            "dfg_edges": z["dfg_edges"],  # (N,) object[np.ndarray(E_i,2)]
            "token_predicate_id": z[
                "token_predicate_id"
            ],  # (N,) object[np.ndarray(L_i,)]
            "predicate_vocab_local": list(
                zip(z["predicate_vocab_local_keys"], z["predicate_vocab_local_vals"])
            ),  # list of (keys[], vals[])
        }


if __name__ == "__main__":
    # Ví dụ CLI tối giản
    import argparse, json as _json
    from transformers import T5TokenizerFast

    parser = argparse.ArgumentParser(
        description="Build DFG-links dataset (.npz) ở mức T5 pieces từ JSON"
    )
    parser.add_argument(
        "--input", default="data/samples.json", help="Đường dẫn file JSON input"
    )
    parser.add_argument(
        "--output",
        default="data/dfg/samples_dfg_links.npz",
        help="Đường dẫn file .npz sẽ được tạo",
    )
    parser.add_argument(
        "--error_log",
        default="data/dfg/samples_error_log.jsonl",
        help="(Tùy chọn) file .errors.jsonl",
    )
    parser.add_argument(
        "--t5",
        default="t5-base",
        help="Tên/đường dẫn tokenizer T5TokenizerFast",
    )
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(args.t5)
    builder = DFGLinksDatasetBuilderT5(tokenizer=tokenizer, max_length=args.max_length)

    stats = builder.build_and_save(args.input, args.output, args.error_log)
    print(_json.dumps(stats, ensure_ascii=False, indent=2))
