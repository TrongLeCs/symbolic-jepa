# parse_fol_file.py
import json
from pathlib import Path
from typing import List, Dict, Any, Union
from parser.fol_parser_api import (
    FOLParser,
    parsed_tree,
)  # nếu đặt chung file thì bỏ import này


def _load_json_as_list(p: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(p)
    raw = p.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON input không hợp lệ: {e}")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("File JSON phải là list (hoặc object đơn).")


def parse_and_dump(
    input_json_path: Union[str, Path],
    output_txt_path: Union[str, Path],
    error_log_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    input_json_path = Path(input_json_path)
    output_txt_path = Path(output_txt_path)

    data = _load_json_as_list(input_json_path)

    out_lines: List[str] = []
    errors: List[Dict[str, Any]] = []

    out_lines.append(f"# FOL parse dump for {input_json_path.name}\n")
    out_lines.append(f"# Total input records: {len(data)}\n")
    out_lines.append("# --------------------------------------------\n\n")

    for idx, rec in enumerate(data):
        topic = rec.get("topic", None)
        nl = rec.get("nl", "")
        fol = rec.get("fol", "")

        if topic is None or not isinstance(fol, str) or fol.strip() == "":
            errors.append(
                {
                    "index_in_file": idx,
                    "topic": topic,
                    "error": "Missing 'topic' or empty 'fol'",
                }
            )
            continue

        try:
            # tokenize để xem
            fol_tok = FOLParser()
            tokens = fol_tok.tokenize(fol)

            # parse ra AST
            fol_parse = FOLParser()
            ast = fol_parse.parse(fol)
            tree = parsed_tree(ast)

            # ghi ra TXT
            out_lines.append(f"=== Record {idx} ===\n")
            out_lines.append(f"topic: {topic}\n")
            out_lines.append(f"nl: {nl}\n")
            out_lines.append(f"fol: {fol}\n")
            # out_lines.append(f"tokens: {tokens}\n")
            out_lines.append("parsed_tree:\n")
            out_lines.append(tree)
            out_lines.append("\n")

        except Exception as e:
            errors.append(
                {
                    "index_in_file": idx,
                    "topic": topic,
                    "fol": fol,
                    "error": f"{type(e).__name__}: {str(e)}",
                }
            )

    # nếu có lỗi, thêm phần lỗi vào cuối file txt cho dễ tra cứu
    if errors:
        out_lines.append("=== ERRORS ===\n")
        for e in errors:
            out_lines.append(json.dumps(e, ensure_ascii=False) + "\n")
        out_lines.append("\n")

    # ghi txt
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text("".join(out_lines), encoding="utf-8")

    # ghi jsonl riêng (tuỳ chọn)
    if errors and error_log_path is not None:
        error_log_path = Path(error_log_path)
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with error_log_path.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return {
        "num_records": len(data),
        "num_ok": len(data) - len(errors),
        "num_errors": len(errors),
        "output_txt": str(output_txt_path),
        "error_log": (
            str(error_log_path) if errors and error_log_path is not None else None
        ),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Đọc JSON {topic,nl,fol}, parse FOL và lưu kết quả ra .txt"
    )
    parser.add_argument(
        "--input",
        required=False,
        default="data/val.json",
        help="Đường dẫn file JSON input (list record).",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="data/trees/val_tree.txt",
        help="Đường dẫn file .txt để ghi kết quả.",
    )
    parser.add_argument(
        "--error_log",
        required=False,
        default="data/trees/val_tree" + ".errors.jsonl",
        help="(Tuỳ chọn) đường dẫn .errors.jsonl để ghi lỗi.",
    )
    args = parser.parse_args()

    stats = parse_and_dump(args.input, args.output, args.error_log)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
