from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Any

Modality = Literal["NL", "FOL"]


@dataclass
class Segment:
    seg_id: int
    modality: Modality  # "NL" | "FOL"
    expression: str
    tokens: List[Tuple[str, int]]  # (string, local_id)
    type_paths: List[Dict] | None
    value_paths: List[Dict] | None


def _shift_paths(
    paths: List[Dict] | None, id_map_local2global: Dict[int, int]
) -> List[Dict]:
    """
    Shift local node IDs to global token IDs within the path dictionaries.
    """
    if not paths:
        return []
    out = []
    for p in paths:
        g = dict(p)
        g["current_id"] = id_map_local2global.get(p["current_id"], p["current_id"])
        out.append(g)
    return out


def _as_list(x: Any) -> List[Dict]:
    """
    Normalize a field that could be None, a dict, or a list of dicts into a list of dicts.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # x is a single item (dict, or rarely another type) -> wrap it in a list
    return [x]


def _norm_tokens(tok_list: Any) -> List[Tuple[str, int]]:
    """
    Normalize a list of token pairs [[tok, id] | (tok, id)] into a list of (str, int) tuples.
    """
    out: List[Tuple[str, int]] = []
    if not tok_list:
        return out
    for t in tok_list:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            tok, lid = t
            out.append((str(tok), int(lid)))
    return out


def _build_text_and_spans(
    tokens: List[Tuple[str, int]],
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Concatenate tokens with a single space to create text and return character-level
    spans (start, end) perfectly aligned 1-to-1 with the token sequence.
    """
    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    cur = 0
    for i, (t, _) in enumerate(tokens):
        if i > 0:
            pieces.append(" ")
            cur += 1
        start = cur
        pieces.append(t)
        cur += len(t)
        spans.append((start, cur))
    return "".join(pieces), spans


def linearize_sample(sample_data: Dict) -> Dict:
    """
    Linearize a data sample by merging Natural Language (NL) and First-Order Logic (FOL) segments.

    Input:
      - ast_nl: List[dict] or None
      - ast_fol: dict (a single item) or List[dict] or None

    Output normalized mixed item:
      - tokens: List[(tok, global_id)]
      - seg_meta: List[(seg_id, modality_int)] where modality_int: 0=NL, 1=FOL
            - type_paths, value_paths: paths with current_id shifted to global_id
      - sentences: List[{
            'seg_id': int,
            'modality': 'NL'|'FOL',
            'text': str,
            'spans': List[(start,end)]        # Aligned with the segment's tokens
            'tok_indices': List[int]          # Corresponding global token indices
        }]
      - L: Total number of global tokens
      - num_segments: Total number of segments
    """
    segments: List[Segment] = []
    seg_id = 0

    # NL (normalized to a list)
    for nl in _as_list(sample_data.get("ast_nl")):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="NL",
                expression=nl.get("expression", ""),
                tokens=_norm_tokens(nl.get("tokens", [])),
                type_paths=nl.get("type_paths", nl.get("leaf", [])),
                value_paths=nl.get("value_paths", nl.get("path", [])),
            )
        )
        seg_id += 1

    # FOL (handles ast_fol as a single item or a list)
    for fol in _as_list(sample_data.get("ast_fol")):
        segments.append(
            Segment(
                seg_id=seg_id,
                modality="FOL",
                expression=fol.get("expression", ""),
                tokens=_norm_tokens(fol.get("tokens", [])),
                type_paths=fol.get("type_paths", fol.get("leaf", [])),
                value_paths=fol.get("value_paths", fol.get("path", [])),
            )
        )
        seg_id += 1

    # Merge tokens and build local->global ID map for each segment
    tokens: List[Tuple[str, int]] = []
    seg_meta: List[Tuple[int, int]] = []  # (seg_id, modality_int)
    type_paths_all: List[Dict] = []
    value_paths_all: List[Dict] = []
    expressions: List[str] = []
    sentences: List[Dict] = []

    gid = 0  # global token index (in order of appearance)
    for seg in segments:
        # map local ID to global ID
        id_map: Dict[int, int] = {}

        # record the global start index of this segment
        start_gid = gid
        for tok, lid in seg.tokens:
            id_map[lid] = gid
            tokens.append((tok, gid))
            seg_meta.append((seg.seg_id, 0 if seg.modality == "NL" else 1))
            gid += 1
        end_gid = gid  # exclusive

        # shift paths to use global IDs
        type_paths_all += _shift_paths(seg.type_paths, id_map)
        value_paths_all += _shift_paths(seg.value_paths, id_map)

        # text & spans for the segment (for sentence-level embedding)
        text, spans = _build_text_and_spans(seg.tokens)
        tok_indices = list(range(start_gid, end_gid)) if (end_gid > start_gid) else []

        sentences.append(
            {
                "seg_id": seg.seg_id,
                "modality": seg.modality,
                "text": text,
                "spans": spans,
                "tok_indices": tok_indices,
            }
        )

        expressions.append(seg.expression)

    return {
        "topic": sample_data.get("topic"),
        "tokens": tokens,
        "seg_meta": seg_meta,
        "type_paths": type_paths_all,
        "value_paths": value_paths_all,
        # Backward compatibility for older downstream code.
        "leaf": type_paths_all,
        "path": value_paths_all,
        "num_segments": len(segments),
        "sentences": sentences,  # used for sentence-level embedding
        "L": len(tokens),
    }
