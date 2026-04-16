# ======= Node type vocab (customize as needed) =======
import re
from typing import List, Tuple

DEFAULT_TYPE_VOCAB = {
    "FORALL": 0,
    "VAR": 1,
    "PRED": 2,
    "IMPLIES": 3,
    "EXISTS": 4,
    "AND": 5,
    "OR": 6,
    "XOR": 7,
    "IFF": 8,
    "NOT": 9,
    "GROUP": 10,
}

OPERATORS = {"AND", "OR", "XOR", "IMPLIES", "IFF", "NOT"}
QUANTIFIERS = {"FORALL", "EXISTS"}
IGNORED_TOKENS = {"(", ")", ","}


def is_identifier(tok: str) -> bool:
    if tok in IGNORED_TOKENS or tok in OPERATORS or tok in QUANTIFIERS:
        return False
    return re.fullmatch(r"[A-Za-z_]\w*", tok) is not None


def normalize_symbols(expression: str) -> str:
    repl = {
        "∀": "FORALL ",
        "∃": "EXISTS ",
        "→": " IMPLIES ",
        "↔": " IFF ",
        "¬": " NOT ",
        "∧": " AND ",
        "∨": " OR ",
        "⊕": " XOR ",
    }
    out = expression
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def fol_tokenize_with_spans(expr: str) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    """
    Tokenize FOL (giữ () ,) và trả về:
      - text_norm: chuỗi normalized để đưa vào T5TokenizerFast (add_special_tokens=False)
      - toks: danh sách token FOL (dạng whitespace-sep, không BOS/EOS)
      - spans: list (char_start, char_end_exclusive) cho từng token trong text_norm
    """
    expr = normalize_symbols(expr)
    expr = expr.replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
    expr = re.sub(r"\s+", " ", expr).strip()
    toks = expr.split() if expr else []

    spans: List[Tuple[int, int]] = []
    pieces = []
    pos = 0
    for tok in toks:
        start = pos
        end = start + len(tok)
        spans.append((start, end))
        pieces.append(tok)
        pos = end + 1  # one space
    text_norm = " ".join(pieces)
    return text_norm, toks, spans
