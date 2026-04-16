# ast_paths_tokenlevel_t5fast.py
# Build AST path labels *at T5TokenizerFast piece level* (1:1 with decoder hidden states)
# - No node pooling
# - Only AST Path (no DFG here)
# - Works with T5TokenizerFast using offset_mapping to align FOL tokens -> subword pieces

from typing import List, Tuple, Dict, Any, Optional
import re
import numpy as np

from parser.common import (
    IGNORED_TOKENS,
    OPERATORS,
    QUANTIFIERS,
    DEFAULT_TYPE_VOCAB,
    fol_tokenize_with_spans,
    is_identifier,
)

try:
    from transformers import T5TokenizerFast
except Exception:
    T5TokenizerFast = None  # for static analysis / import in environments without HF

# ===== Normalize & basic FOL tokenization with spans (for alignment) =====


# ===== Minimal AST builder (no '.') =====
class Node:
    __slots__ = (
        "kind",
        "start",
        "end",
        "children",
        "pred_name_idx",
        "arg_indices",
        "quant_type",
        "bound_var_idx",
        "anchor",
    )

    def __init__(self, kind: str, start: int, end: int):
        self.kind = kind
        self.start = start
        self.end = end
        self.children: List["Node"] = []
        self.pred_name_idx: Optional[int] = None
        self.arg_indices: List[int] = []
        self.quant_type: Optional[str] = None
        self.bound_var_idx: Optional[int] = None
        self.anchor: Optional[int] = None


class TokStream:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0
        self.n = len(tokens)

    def peek(self, k: int = 0) -> Optional[str]:
        j = self.i + k
        return self.toks[j] if 0 <= j < self.n else None

    def take(self) -> Optional[str]:
        if self.i < self.n:
            t = self.toks[self.i]
            self.i += 1
            return t
        return None

    def idx(self) -> int:
        return self.i


def find_matching_paren(tokens: List[str], open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(tokens)):
        if tokens[i] == "(":
            depth += 1
        elif tokens[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_expression(tokens: List[str]) -> Node:
    ts = TokStream(tokens)
    node = parse_iff(ts, tokens)
    if ts.i != ts.n:
        raise ValueError(f"Unexpected tail tokens: {tokens[ts.i:]}")
    return node


def parse_iff(ts: TokStream, T: List[str]) -> Node:
    left = parse_implies(ts, T)
    while ts.peek() == "IFF":
        op_idx = ts.idx()
        ts.take()
        right = parse_implies(ts, T)
        n = Node(
            "IFF",
            min(left.start, op_idx, right.start),
            max(left.end, op_idx, right.end),
        )
        n.children = [left, right]
        n.anchor = op_idx
        left = n
    return left


def parse_implies(ts: TokStream, T: List[str]) -> Node:
    left = parse_or(ts, T)
    while ts.peek() == "IMPLIES":
        op_idx = ts.idx()
        ts.take()
        right = parse_or(ts, T)
        n = Node(
            "IMPLIES",
            min(left.start, op_idx, right.start),
            max(left.end, op_idx, right.end),
        )
        n.children = [left, right]
        n.anchor = op_idx
        left = n
    return left


def parse_or(ts: TokStream, T: List[str]) -> Node:
    left = parse_and(ts, T)
    while ts.peek() == "OR":
        op_idx = ts.idx()
        ts.take()
        right = parse_and(ts, T)
        n = Node(
            "OR", min(left.start, op_idx, right.start), max(left.end, op_idx, right.end)
        )
        n.children = [left, right]
        n.anchor = op_idx
        left = n
    return left


def parse_and(ts: TokStream, T: List[str]) -> Node:
    left = parse_xor(ts, T)
    while ts.peek() == "AND":
        op_idx = ts.idx()
        ts.take()
        right = parse_xor(ts, T)
        n = Node(
            "AND",
            min(left.start, op_idx, right.start),
            max(left.end, op_idx, right.end),
        )
        n.children = [left, right]
        n.anchor = op_idx
        left = n
    return left


def parse_xor(ts: TokStream, T: List[str]) -> Node:
    left = parse_quantifier_or_not(ts, T)
    while ts.peek() == "XOR":
        op_idx = ts.idx()
        ts.take()
        right = parse_quantifier_or_not(ts, T)
        n = Node(
            "XOR",
            min(left.start, op_idx, right.start),
            max(left.end, op_idx, right.end),
        )
        n.children = [left, right]
        n.anchor = op_idx
        left = n
    return left


def parse_quantifier_or_not(ts: TokStream, T: List[str]) -> Node:
    tok = ts.peek()
    if tok in QUANTIFIERS:
        qtype = tok
        q_idx = ts.idx()
        ts.take()
        var_tok = ts.peek()
        if var_tok is None or not is_identifier(var_tok):
            raise ValueError("Expected variable after quantifier")
        var_idx = ts.idx()
        ts.take()
        body = parse_quantifier_or_not(ts, T)
        n = Node(qtype, q_idx, body.end)
        n.quant_type = qtype
        n.bound_var_idx = var_idx
        n.children = [body]
        n.anchor = q_idx
        return n
    if tok == "NOT":
        op_idx = ts.idx()
        ts.take()
        sub = parse_atom(ts, T)
        n = Node("NOT", min(op_idx, sub.start), max(op_idx, sub.end))
        n.children = [sub]
        n.anchor = op_idx
        return n
    return parse_atom(ts, T)


def parse_atom(ts: TokStream, T: List[str]) -> Node:
    tok = ts.peek()
    if tok == "(":
        l = ts.idx()
        ts.take()
        sub = parse_iff(ts, T)
        if ts.peek() != ")":
            raise ValueError("Expected ')'")
        r = ts.idx()
        ts.take()
        n = Node("GROUP", l, r)
        n.children = [sub]
        return n
    if tok is not None and is_identifier(tok) and ts.peek(1) == "(":
        name_idx = ts.idx()
        ts.take()
        l = ts.idx()
        ts.take()
        r = find_matching_paren(T, l)
        if r == -1:
            raise ValueError("Unbalanced parentheses in predicate call")
        arg_indices = []
        j = l + 1
        while j < r:
            if T[j] not in IGNORED_TOKENS and is_identifier(T[j]):
                arg_indices.append(j)
            j += 1
        ts.i = r + 1
        n = Node("PRED", name_idx, r)
        n.pred_name_idx = name_idx
        n.arg_indices = arg_indices
        return n
    if tok is not None and is_identifier(tok):
        idx = ts.idx()
        ts.take()
        return Node("VAR", idx, idx)
    raise ValueError(f"Unexpected token {tok} at position {ts.idx()}")


def collect_paths_for_tokens(
    tokens: List[str],
    root: Node,
    type_vocab: Dict[str, int],
    max_depth: int,
) -> np.ndarray:
    L = len(tokens)
    paths = np.full((1, L, max_depth), -1, dtype=np.int64)
    token_paths: List[List[Tuple[str, int]]] = [[] for _ in range(L)]

    def mark_token(idx: int, labels: List[Tuple[str, int]]):
        if 0 <= idx < L:
            token_paths[idx] = labels

    def push_anc(anc, label, anchor_idx):
        return anc + [(label, anchor_idx)]

    def dfs(node: Node, anc: List[Tuple[str, int]]):
        kind = node.kind
        if kind in QUANTIFIERS or kind in OPERATORS:
            anchor_idx = node.anchor if node.anchor is not None else node.start
            anc_next = push_anc(anc, kind, anchor_idx)
            mark_token(anchor_idx, anc + [(kind, anchor_idx)])
        elif kind == "GROUP":
            anc_next = anc
        else:
            anc_next = anc

        if kind == "PRED":
            if node.pred_name_idx is not None:
                labels = anc_next + [("PRED", node.pred_name_idx)]
                mark_token(node.pred_name_idx, labels)
            for ai in node.arg_indices:
                labels = anc_next + [("PRED", node.pred_name_idx), ("VAR", ai)]
                mark_token(ai, labels)
        elif kind == "VAR":
            mark_token(node.start, anc_next + [("VAR", node.start)])

        if kind in QUANTIFIERS and node.bound_var_idx is not None:
            mark_token(node.bound_var_idx, anc_next + [("VAR", node.bound_var_idx)])

        for ch in node.children:
            dfs(ch, anc_next)

    dfs(root, [])

    for i in range(L):
        labels = token_paths[i]
        if not labels:
            continue
        if len(labels) > max_depth:
            labels = labels[-max_depth:]
        ids = [type_vocab.get(lab, -1) for (lab, _) in labels]
        for d, v in enumerate(ids):
            paths[0, i, d] = v
    return paths


# ======= Alignment: FOL tokens -> T5 pieces via offsets =======


def align_t5_pieces_to_fol(
    spans_fol, piece_offsets, text_norm, piece_tokens=None, ignore_whitespace=True
):
    out = []
    for j, (ps, pe) in enumerate(piece_offsets):
        if ps == pe:
            out.append(-1)
            continue
        if ignore_whitespace:
            seg = text_norm[ps:pe]
            if (
                piece_tokens is not None and piece_tokens[j] == "▁"
            ) or seg.strip() == "":
                out.append(-1)
                continue

        # chứa hoàn toàn
        idx = -1
        for k, (fs, fe) in enumerate(spans_fol):
            if ps >= fs and pe <= fe:
                idx = k
                break
        if idx == -1:
            # chọn theo overlap lớn nhất
            best_k, best_ol = -1, -1
            for k, (fs, fe) in enumerate(spans_fol):
                ol = max(0, min(pe, fe) - max(ps, fs))
                if ol > best_ol:
                    best_ol, best_k = ol, k
            # nếu overlap == 0 với mọi span -> coi là khoảng trắng
            idx = best_k if best_ol > 0 else -1

        out.append(idx)
    return out


# ======= Public API: build AST paths at T5 piece-level =======


def build_ast_paths_tokenlevel(
    expr: str,
    tokenizer: str,
    max_depth: int = 8,
    max_length: int = 256,
    type_vocab: Dict[str, int] = None,
) -> Dict[str, Any]:
    """
    Return dict with the following keys:
      - text_norm: str (input fed to tokenizer with add_special_tokens=False)
      - fol_tokens: List[str]
      - fol_spans: List[(start,end)] in text_norm
      - input_ids: List[int] (T5 pieces, no specials)
      - attention_mask: List[int]
      - offsets: List[(start,end)] per piece (same space as text_norm)
      - piece2fol: List[int] mapping each piece -> fol token idx
      - ast_paths_fol: np.ndarray shape (1, L_fol, max_depth)
      - ast_paths_t5:  np.ndarray shape (1, L_piece, max_depth)  (duplicated per piece)
    """
    if type_vocab is None:
        type_vocab = DEFAULT_TYPE_VOCAB

    # 1) Normalize & FOL tokens + spans
    text_norm, fol_tokens, fol_spans = fol_tokenize_with_spans(expr)

    # 2) Parse over FOL tokens and collect paths (at FOL-token level)
    root = parse_expression(fol_tokens)
    ast_paths_fol = collect_paths_for_tokens(
        fol_tokens, root, type_vocab, max_depth
    )

    # 3) Tokenize with T5 fast to get *piece*-level ids & offsets
    enc = tokenizer(
        text_norm,
        max_length=max_length,
        truncation=True,
        padding="longest",
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"]  # list of (start,end) in text_norm

    # 4) Align pieces -> FOL tokens
    piece_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    piece2fol = align_t5_pieces_to_fol(
        fol_spans, offsets, text_norm, piece_tokens=piece_tokens, ignore_whitespace=True
    )

    # 5) Project AST paths onto piece level (duplicate labels for pieces of same FOL token)
    L_piece = len(input_ids)
    ast_paths_t5 = np.full((1, L_piece, max_depth), -1, dtype=np.int64)
    for j in range(L_piece):
        k = piece2fol[j]
        if k is None or k < 0:  # unmatched (e.g., zero-length piece)
            continue
        ast_paths_t5[0, j, :] = ast_paths_fol[0, k, :]

    return {
        "text_norm": text_norm,
        "fol_tokens": fol_tokens,
        "fol_spans": fol_spans,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offsets": offsets,
        "piece2fol": piece2fol,
        "ast_paths_fol": ast_paths_fol,
        "ast_paths_t5": ast_paths_t5,
    }


# ======= Example usage =======
if __name__ == "__main__":
    # Requires: pip install transformers sentencepiece
    # and a tokenizer, e.g., t5-small
    name = "t5-small"
    if T5TokenizerFast is None:
        print("Transformers not available; skip demo.")
    else:
        tok = T5TokenizerFast.from_pretrained(name)
        expr = "(FORALL x (item(x) AND break_easy_stress_impact(x) IMPLIES fragile(x)))"
        out = build_ast_paths_tokenlevel(expr, tok, max_depth=6)
        print("text_norm:", out["text_norm"])
        print("fol_tokens:", out["fol_tokens"])
        print("fol_spans:", out["fol_spans"])
        print("input_ids:", out["input_ids"][:32])
        print("offsets   :", out["offsets"][:32])
        print("piece2fol :", out["piece2fol"][:32])
        print("ast_paths_fol shape:", out["ast_paths_fol"].shape)
        print("ast_paths_t5  shape:", out["ast_paths_t5"].shape)
