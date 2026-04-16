from typing import List, Tuple, Dict, Any, Set, Optional
import re
import numpy as np

from parser.common import (
    IGNORED_TOKENS,
    OPERATORS,
    QUANTIFIERS,
    fol_tokenize_with_spans,
    is_identifier,
)


def find_matching_paren(tokens: List[str], open_idx: int, end_limit: int) -> int:
    depth = 0
    for i in range(open_idx, end_limit + 1):
        if tokens[i] == "(":
            depth += 1
        elif tokens[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def is_flat_arg_list(tokens: List[str], l: int, r: int) -> bool:
    depth = 0
    for i in range(l, r + 1):
        t = tokens[i]
        if t == "(":
            depth += 1
            if depth > 1:
                return False
        elif t == ")":
            depth -= 1
        elif t in OPERATORS or t in QUANTIFIERS:
            return False
    return True


def scan_predicates(tokens: List[str]) -> List[Dict[str, Any]]:
    preds = []
    N = len(tokens)
    i = 0
    while i <= N - 1:
        if is_identifier(tokens[i]) and i + 1 < N and tokens[i + 1] == "(":
            name_idx = i
            l = i + 1
            r = find_matching_paren(tokens, l, N - 1)
            if r == -1:
                i += 1
                continue
            if not is_flat_arg_list(tokens, l, r):
                i += 1
                continue
            args_idx = []
            j = l + 1
            while j < r:
                if tokens[j] not in IGNORED_TOKENS and is_identifier(tokens[j]):
                    args_idx.append(j)
                j += 1
            preds.append({"name_idx": name_idx, "args_idx": args_idx, "span": (i, r)})
            i = r + 1
        else:
            i += 1
    return preds


def scan_quantifiers(tokens: List[str]) -> List[Dict[str, Any]]:
    quants = []
    N = len(tokens)
    i = 0
    while i < N:
        if tokens[i] in QUANTIFIERS and i + 1 < N:
            q_idx = i
            var_idx = i + 1
            var_name = tokens[var_idx]
            scope_start = var_idx + 1
            scope_end = N - 1
            if scope_start <= N - 1 and tokens[scope_start] == "(":
                scope_end = find_matching_paren(tokens, scope_start, N - 1)
                if scope_end == -1:
                    scope_end = N - 1
            else:
                if scope_start <= N - 1 and is_identifier(tokens[scope_start]):
                    if scope_start + 1 <= N - 1 and tokens[scope_start + 1] == "(":
                        r = find_matching_paren(tokens, scope_start + 1, N - 1)
                        scope_end = r if r != -1 else N - 1
                    else:
                        scope_end = scope_start
                else:
                    scope_end = N - 1
            quants.append(
                {
                    "q_idx": q_idx,
                    "var_idx": var_idx,
                    "var_name": var_name,
                    "scope": (scope_start, scope_end),
                }
            )
            i = scope_end + 1
        else:
            i += 1
    return quants


def tag_token_types(tokens: List[str]) -> List[str]:
    types = ["other"] * len(tokens)
    for idx, t in enumerate(tokens):
        if t in IGNORED_TOKENS:
            types[idx] = "ignored"
        elif t in QUANTIFIERS:
            types[idx] = "quantifier"
        elif t in OPERATORS:
            types[idx] = "operator"
    for p in scan_predicates(tokens):
        types[p["name_idx"]] = "predicate"
        for a in p["args_idx"]:
            if types[a] == "other":
                types[a] = "variable"
    for q in scan_quantifiers(tokens):
        if types[q["var_idx"]] == "other":
            types[q["var_idx"]] = "variable"
    for i, t in enumerate(tokens):
        if types[i] == "other" and is_identifier(t):
            types[i] = "variable"
    return types


def build_edges(tokens: List[str]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for p in scan_predicates(tokens):
        s = p["name_idx"]
        for a in p["args_idx"]:
            edges.append((s, a))
    for q in scan_quantifiers(tokens):
        s = q["q_idx"]
        var = q["var_name"]
        qs, qe = q["scope"]
        for i in range(max(0, qs), min(len(tokens) - 1, qe) + 1):
            if tokens[i] == var and i != q["var_idx"]:
                edges.append((s, i))
    return edges


def build_ldp_matrix(
    N: int, edges: List[Tuple[int, int]], ignored_indices: List[int]
) -> np.ndarray:
    mat = np.zeros((1, N, N), dtype=np.int8)
    for i, j in edges:
        if 0 <= i < N and 0 <= j < N:
            mat[0, i, j] = 1
    for idx in ignored_indices:
        mat[0, idx, :] = -1
        mat[0, :, idx] = -1
    return mat


def build_predicate_vocab(tokens: List[str]) -> Tuple[Dict[str, int], List[int]]:
    vocab: Dict[str, int] = {}
    next_id = 0
    token_pred_id = [-1] * len(tokens)
    for p in scan_predicates(tokens):
        name_tok = tokens[p["name_idx"]]
        if name_tok not in vocab:
            vocab[name_tok] = next_id
            next_id += 1
        token_pred_id[p["name_idx"]] = vocab[name_tok]
    return vocab, token_pred_id


# =========================
# 3) Căn chỉnh T5 (skip whitespace pieces) + chiếu nhãn/LDP
# =========================
def t5_tokenize(
    text_norm: str, tokenizer, max_length
) -> Tuple[List[int], List[str], List[Tuple[int, int]], List[int]]:
    enc = tokenizer(
        text_norm,
        max_length=int(max_length),
        padding="longest",  # hoặc "max_length"
        truncation=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = [(int(s), int(e)) for (s, e) in enc["offset_mapping"]]
    pieces = tokenizer.convert_ids_to_tokens(input_ids)
    return input_ids, pieces, offsets, attn


def align_t5_pieces_to_fol(
    fol_spans: List[Tuple[int, int]],
    piece_offsets: List[Tuple[int, int]],
    text_norm: str,
    piece_tokens: List[str],
    ignore_whitespace: bool = True,
) -> Tuple[List[int], List[List[int]]]:
    """
    Trả về:
      - piece2fol: List[int] (=-1 nếu không khớp / bị bỏ qua)
      - fol2piece: List[List[int]] chỉ gồm các piece hợp lệ (đÃ bỏ '▁')
    Quy tắc: ưu tiên containment, fallback theo intersection dài nhất.
    """

    def is_ws_span(span: Tuple[int, int]) -> bool:
        s, e = span
        if e <= s:
            return True
        return text_norm[s:e].strip() == ""

    def inter_len(a, b):
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return max(0, e - s)

    piece2fol: List[int] = [-1] * len(piece_offsets)
    fol2piece: List[List[int]] = [[] for _ in range(len(fol_spans))]

    for pi, pspan in enumerate(piece_offsets):
        tok = piece_tokens[pi]
        # BỎ pure space '▁', offset rỗng, hoặc span toàn whitespace
        if (
            tok == "▁"
            or pspan[0] == pspan[1]
            or (ignore_whitespace and is_ws_span(pspan))
        ):
            continue

        # 1) containment
        assigned = -1
        for fi, fspan in enumerate(fol_spans):
            if fspan[0] <= pspan[0] and pspan[1] <= fspan[1]:
                piece2fol[pi] = fi
                fol2piece[fi].append(pi)
                assigned = fi
                break
        if assigned != -1:
            continue

        # 2) fallback: max intersection
        best_fi, best_il = -1, 0
        for fi, fspan in enumerate(fol_spans):
            il = inter_len(pspan, fspan)
            if il > best_il:
                best_il = il
                best_fi = fi
        if best_fi != -1 and best_il > 0:
            piece2fol[pi] = best_fi
            fol2piece[best_fi].append(pi)

    return piece2fol, fol2piece


def pick_head_piece(fol2piece: List[List[int]]) -> List[Optional[int]]:
    """Chọn subtoken đầu (đầu tiên hợp lệ) cho mỗi FOL token; None nếu không có."""
    head = [p[0] if len(p) > 0 else None for p in fol2piece]
    return head


def project_edges_to_t5(
    edges_fol: List[Tuple[int, int]],
    fol2piece: List[List[int]],
    mode: str = "first",  # "first" | "all"
) -> List[Tuple[int, int]]:
    edges_t5: List[Tuple[int, int]] = []
    if mode == "first":
        head = pick_head_piece(fol2piece)
        for i_f, j_f in edges_fol:
            si = head[i_f]
            sj = head[j_f]
            if si is not None and sj is not None:
                edges_t5.append((si, sj))
    else:
        for i_f, j_f in edges_fol:
            srcs = fol2piece[i_f]
            dsts = fol2piece[j_f]
            for si in srcs:
                for sj in dsts:
                    edges_t5.append((si, sj))
    return edges_t5


# =========================
# 4) API chính: LDP (FOL & T5)
# =========================
def build_ldp_links_tokenlevel(expr: str, tokenizer, max_length) -> Dict[str, Any]:
    """
    Xây LDP ở mức FOL và chiếu sang mức T5 pieces (đã lọc whitespace pieces).
    """
    # FOL
    text_norm, fol_tokens, fol_spans = fol_tokenize_with_spans(expr)
    types_fol = tag_token_types(fol_tokens)
    edges_fol = build_edges(fol_tokens)
    ignored_fol_idx = [i for i, t in enumerate(fol_tokens) if t in IGNORED_TOKENS]
    ldp_fol = build_ldp_matrix(len(fol_tokens), edges_fol, ignored_fol_idx)
    predicate_vocab, token_predicate_id_fol = build_predicate_vocab(fol_tokens)

    # T5
    input_ids, piece_tokens, offsets, attention_mask = t5_tokenize(
        text_norm, tokenizer, max_length
    )
    piece2fol, fol2piece = align_t5_pieces_to_fol(
        fol_spans, offsets, text_norm, piece_tokens=piece_tokens, ignore_whitespace=True
    )

    # types & predicate ids trên piece
    types_t5 = ["other"] * len(input_ids)
    token_predicate_id_t5 = [-1] * len(input_ids)

    # Gán type theo mapping piece->FOL
    for pi, fi in enumerate(piece2fol):
        if fi >= 0:
            types_t5[pi] = types_fol[fi]

    for fi, pid in enumerate(token_predicate_id_fol):
        if pid != -1:  # fi là vị trí FOL-token tên predicate
            for pi in fol2piece[fi]:  # mọi piece thuộc FOL-token đó
                token_predicate_id_t5[pi] = pid

    edges_t5 = project_edges_to_t5_smart(edges_fol, fol2piece, types_fol)

    # ignored indices trên piece: (1) piece2fol=-1 (whitespace/unmatched) hoặc (2) map sang FOL ignored
    ignored_t5_idx = []
    for pi, fi in enumerate(piece2fol):
        if piece_tokens[pi] == "▁" or fi < 0:
            ignored_t5_idx.append(pi)
        elif fol_tokens[fi] in IGNORED_TOKENS:
            ignored_t5_idx.append(pi)

    ldp_t5 = build_ldp_matrix(len(input_ids), edges_t5, ignored_t5_idx)

    return {
        "text_norm": text_norm,
        "fol_tokens": fol_tokens,
        "fol_spans": fol_spans,
        "types_fol": types_fol,
        "ldp_links_fol": ldp_fol,
        "predicate_vocab": predicate_vocab,
        "token_predicate_id_fol": np.array(token_predicate_id_fol, dtype=np.int32),
        "input_ids": np.array(input_ids, dtype=np.int32),
        "t5_tokens": piece_tokens,
        "offsets": offsets,
        "attention_mask": attention_mask,
        "piece2fol": np.array(piece2fol, dtype=np.int32),
        "fol2piece": fol2piece,
        "types_t5": types_t5,
        "token_predicate_id_t5": np.array(token_predicate_id_t5, dtype=np.int32),
        "ldp_links_t5": ldp_t5,
    }


def project_edges_to_t5_smart(
    edges_fol: List[Tuple[int, int]],
    fol2piece: List[List[int]],
    types_fol: List[str],
) -> List[Tuple[int, int]]:
    """
    Chiếu cạnh FOL -> T5 pieces.

    - Nguồn:
        * predicate  -> ALL subtokens
        * quantifier -> head
        * khác       -> head
    - ĐÍCH: ALL subtokens (cố định)
    """
    head: List[Optional[int]] = pick_head_piece(fol2piece)
    edges_t5: List[Tuple[int, int]] = []

    def src_indices(fi: int) -> List[int]:
        if types_fol[fi] == "predicate":
            # nguồn là tên predicate -> dùng toàn bộ subtokens
            return fol2piece[fi]
        # quantifier/khác -> head nếu có
        return [] if head[fi] is None else [head[fi]]

    def dst_indices(fi: int) -> List[int]:
        # luôn dùng ALL subtokens làm đích
        return list(fol2piece[fi])

    for i_f, j_f in edges_fol:
        for si in src_indices(i_f):
            for sj in dst_indices(j_f):
                edges_t5.append((si, sj))
    return edges_t5


# =========================
# Demo
# =========================
if __name__ == "__main__":
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained("tokenizers-extended")
    expr = "(FORALL x (mirror(x) IMPLIES reflect_light(x))) AND ( FORALL x (reflect_light(x) IMPLIES create_image(x)))"

    graph = build_ldp_links_tokenlevel(
        expr=expr, tokenizer=tokenizer, max_length=256
    )

    # In kết quả
    print("=== TEXT NORM ===")
    print(graph["text_norm"])

    print("\n=== FOL TOKENS ===")
    fol_tokens = graph["fol_tokens"]
    for i, t in enumerate(fol_tokens):
        print(
            f"{i:2d}: {t:18s} type={graph['types_fol'][i]:12s} span={graph['fol_spans'][i]}  pred_id={graph['token_predicate_id_fol'][i]}"
        )

    print("\n=== T5 PIECES (decoder-compatible) ===")
    t5_tokens = graph["t5_tokens"]
    offsets = graph["offsets"]
    piece2fol = graph["piece2fol"]
    types_t5 = graph["types_t5"]
    token_predicate_id_t5 = graph["token_predicate_id_t5"]
    for i, tok in enumerate(t5_tokens):
        fol_idx = int(piece2fol[i])
        print(
            f"{i:2d}: {tok:18s} off={offsets[i]}  map_fol={fol_idx:2d}  type={types_t5[i]:12s}  pred_id={int(token_predicate_id_t5[i])}"
        )

    # LDP (FOL)
    print("\n=== LDP (FOL) edges where value==1 ===")
    ldp_fol = graph["ldp_links_fol"]
    pairs_fol = np.argwhere(ldp_fol[0] == 1)
    for i, j in pairs_fol:
        print(f"{i} -> {j}   ({fol_tokens[i]} -> {fol_tokens[j]})")
    print("LDP (FOL) Matrix shape:", ldp_fol.shape)

    # LDP (T5) — đã lọc whitespace pieces khi align
    print("\n=== LDP (T5) edges where value==1 (projected, first-subtoken) ===")
    ldp_t5 = graph["ldp_links_t5"]
    pairs_t5 = np.argwhere(ldp_t5[0] == 1)
    for i, j in pairs_t5:
        print(f"{i} -> {j}   ({t5_tokens[i]} -> {t5_tokens[j]})")
    print("LDP (T5) Matrix shape:", ldp_t5.shape)

    print("\nPredicate vocab:", graph["predicate_vocab"])
