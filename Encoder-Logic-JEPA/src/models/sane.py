from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast


# =========================== Config ===========================


@dataclass
class SANEConfig:
    t5_name: str = "t5-base"
    d_ast: int = 768
    max_seq_len: int = 512  # For sentence/segment
    max_depth: int = 128
    dropout: float = 0.1
    max_segments: int = 10
    node_type_vocab: Tuple[str, ...] = (
        "FORALL",
        "EXISTS",
        "IFF",
        "IMPLIES",
        "OR",
        "AND",
        "XOR",
        "NOT",
        "Variable",
        "Predicate",
        "NLToken",
    )

    # Control flags for Ablation Study
    use_compositional_path: bool = True  # Enable/disable Leaf and Path features
    use_symbolic_feature: bool = (
        True  # Enable/disable Pos, Depth, Type, Segment, and Modality features
    )

    # T5 for LEAF/PATH in read-only mode
    paths_max_len: int = 64  # Paths are usually short
    paths_chunk: int = 128  # Internal batch size when encoding paths


# =========================== Helpers ===========================


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate the number of overlapping characters between two intervals [a0, a1) and [b0, b1).
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _pool_hidden(h: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Pool the hidden states of subwords into a single representation based on the specified method.
    """
    # h: [n_subwords, d]
    if h.numel() == 0:
        return h.new_zeros((1, h.size(-1)))
    if method == "first":
        return h[0:1]
    if method == "max":
        return h.max(dim=0, keepdim=True).values
    return h.mean(dim=0, keepdim=True)  # Default is mean


class Affine1D(nn.Module):
    """y = x * w + b (element-wise), initialized as identity."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w + self.b


# -------- Context-level per sentence (NL/FOL) --------
class SentenceContextEmbedder(nn.Module):
    """
    Encode a sentence/expression using the T5 encoder, then pool based on
    the character span of each original token.
    """

    def __init__(
        self,
        t5_encoder,
        tokenizer: T5TokenizerFast,
        d_model: int,
        max_len: int = 512,
        pool: str = "mean",
        finetune: bool = True,
    ):
        super().__init__()
        self.t5 = t5_encoder
        self.tk = tokenizer
        self.d = d_model
        self.max_len = max_len
        self.pool = pool
        self.finetune = finetune

    def forward_batch(self, texts: List[str], all_spans: List[List[Tuple[int, int]]]) -> List[torch.Tensor]:
        """
        Forward pass to encode a batch of text strings and pool their token spans.
        Returns: List of [n_tokens_in_sentence, d] tensors.
        """
        if not texts:
            return []
            
        dev = next(self.t5.parameters()).device
        enc = self.tk(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )
        offsets_batch = enc.pop("offset_mapping").tolist()  # [B, S, 2]
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

        ctx = torch.enable_grad() if self.finetune else torch.no_grad()
        with ctx:
            out = self.t5(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).last_hidden_state  # [B, S, d]

        embs_batch = []
        for b, spans in enumerate(all_spans):
            hs = out[b]  # [S, d]
            offsets = offsets_batch[b]
            
            embs_list = []
            for i, sp in enumerate(spans):
                idxs = [j for j, (a, end) in enumerate(offsets) if _overlap(sp, (a, end)) > 0]
                pooled = (
                    _pool_hidden(hs[idxs], "mean") if idxs else hs.new_zeros((1, self.d))
                )
                embs_list.append(pooled[0])

            # Use torch.stack to preserve the computation graph (grad_fn) for backpropagation
            embs = torch.stack(embs_list) if embs_list else hs.new_empty((0, self.d))
            embs_batch.append(embs)
            
        return embs_batch


# -------- Batch T5 embedder for LEAF/PATH (read-only) --------
class T5BatchTextEmbedder(nn.Module):
    """
    Encode a list of strings using the T5 encoder, temporarily using no_grad + eval().
    Returns [N, d].
    """

    def __init__(
        self,
        t5_encoder,
        tokenizer: T5TokenizerFast,
        d_model: int,
        max_len: int = 64,
        chunk: int = 128,
    ):
        super().__init__()
        self.t5 = t5_encoder
        self.tk = tokenizer
        self.d = d_model
        self.max_len = max_len
        self.chunk = chunk

    def forward(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("empty texts")
        dev = next(self.t5.parameters()).device
        out_buf = []

        was_training = self.t5.training
        self.t5.eval()
        with torch.no_grad():
            for s in range(0, len(texts), self.chunk):
                batch_txt = texts[s : s + self.chunk]
                enc = self.tk(
                    batch_txt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                hs = self.t5(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                ).last_hidden_state  # [B, S, d]
                # Mean mask pooling
                mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, S, 1]
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1.0)  # [B, d]
                out_buf.append(pooled.detach())
        if was_training:
            self.t5.train()

        return torch.cat(out_buf, dim=0)  # [N, d]


# ======================= Structure-aware Node & Path Encoder (SANE) =======================


class SANE(nn.Module):
    """
    Structure-Aware Node and Path Encoder (SANE).
    Extracts contextual, compositional, and symbolic features from NL and FOL inputs
    to form the initial node embeddings H(0).
    """

    def __init__(
        self,
        cfg: SANEConfig,
        tokenizer: T5TokenizerFast,
        t5_model: T5ForConditionalGeneration,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.t5_full = t5_model
        self.t5 = (
            t5_model.encoder
        )  # Encoder used for both sentences and paths (paths use no_grad)

        d_t5 = self.t5.config.d_model
        assert d_t5 == cfg.d_ast, f"d_ast ({cfg.d_ast}) must be equal to d_t5 ({d_t5})."
        self._finetune = True

        # Main semantic channel and Compositional Path
        self.aff_context = Affine1D(cfg.d_ast)
        self.aff_leaf = Affine1D(cfg.d_ast)
        self.aff_path = Affine1D(cfg.d_ast)

        # Gates
        self.g_context = nn.Parameter(torch.tensor(1.0))
        self.g_leaf = nn.Parameter(torch.tensor(0.0 if not cfg.use_compositional_path else 1.0))
        self.g_path = nn.Parameter(torch.tensor(0.0 if not cfg.use_compositional_path else 1.0))

        # Symbolic Features (Auxiliary embeddings)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_ast)
        self.depth_emb = nn.Embedding(cfg.max_depth + 1, cfg.d_ast)
        self.ntype2id: Dict[str, int] = {
            n: i for i, n in enumerate(cfg.node_type_vocab)
        }
        self.node_type_emb = nn.Embedding(len(self.ntype2id), cfg.d_ast)
        self.seg_emb = nn.Embedding(cfg.max_segments, cfg.d_ast)
        self.modality_emb = nn.Embedding(2, cfg.d_ast)  # 0=NL, 1=FOL

        self.dropout = nn.Dropout(cfg.dropout)
        self.ln = nn.LayerNorm(cfg.d_ast)

        # Embedders
        self.sent_embedder = SentenceContextEmbedder(
            t5_encoder=self.t5,
            tokenizer=self.tokenizer,
            d_model=cfg.d_ast,
            max_len=cfg.max_seq_len,
            pool="mean",
            finetune=self._finetune,
        )
        self.path_text_embedder = T5BatchTextEmbedder(
            t5_encoder=self.t5,
            tokenizer=self.tokenizer,
            d_model=cfg.d_ast,
            max_len=cfg.paths_max_len,
            chunk=cfg.paths_chunk,
        )

    # --------------- Forward ---------------
    def set_finetune(self, finetune: bool):
        """
        Enable or disable fine-tuning for the context sentence embedder.
        """
        self._finetune = bool(finetune)
        self.sent_embedder.finetune = self._finetune

    def forward(self, item: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute initial structural embeddings.
        """
        device = next(self.parameters()).device

        tokens: List[Tuple[str, int]] = item["tokens"]
        L = len(tokens)
        seg_meta: List[Tuple[int, int]] = item.get("seg_meta", [(0, 1)] * L)
        sentences: List[Dict] = item.get("sentences", [])
        assert (
            sentences
        ), "Missing 'sentences' from mixed_linearizer. Please use the corrected mixed_linearizer version."

        # Map id -> path entry (prefer new keys, keep old-key fallback)
        type_paths_src = item.get("type_paths", item.get("leaf", []))
        value_paths_src = item.get("value_paths", item.get("path", []))
        type_paths = {p["current_id"]: p for p in type_paths_src}
        value_paths = {p["current_id"]: p for p in value_paths_src}

        mask_flags_list = item.get("mask_flags", [0] * L)
        assert len(mask_flags_list) == L, "mask_flags length mismatch"
        mask_flags = torch.tensor(mask_flags_list, device=device, dtype=torch.long)

        pos_ids = torch.arange(L, device=device).clamp(max=self.cfg.max_seq_len - 1)
        seg_ids = torch.tensor([s for s, _ in seg_meta], device=device).clamp(
            max=self.cfg.max_segments - 1
        )
        mod_ids = torch.tensor([m for _, m in seg_meta], device=device)

        # semantic_mask: 1 if the token has semantic meaning (not a parenthesis/bracket)
        semantic_mask = torch.tensor(
            [0 if tok in {"(", ")"} else 1 for tok, _ in tokens],
            device=device,
            dtype=torch.long,
        )

        # ==== 1. Context Encoder (CONTEXT) ====
        token_embeddings = torch.zeros(
            (L, self.cfg.d_ast), device=device, dtype=torch.float32
        )
        
        # Gather all sentences for batched T5 encoding
        texts = []
        all_spans = []
        all_tok_idx = []
        for sent in sentences:
            if not sent["spans"] or not sent["tok_indices"]:
                continue
            texts.append(sent["text"])
            all_spans.append(sent["spans"])
            all_tok_idx.append(sent["tok_indices"])
            
        if texts:
            local_embs = self.sent_embedder.forward_batch(texts, all_spans)
            for b_idx in range(len(texts)):
                local_emb = local_embs[b_idx]
                tok_idx = all_tok_idx[b_idx]
                if local_emb.size(0) != len(tok_idx):
                    n = min(local_emb.size(0), len(tok_idx))
                    local_emb = local_emb[:n]
                    tok_idx = tok_idx[:n]

                # Use index_put (out-of-place) to preserve gradients flowing back to T5
                tok_idx_tensor = torch.as_tensor(tok_idx, device=device, dtype=torch.long)
                token_embeddings = token_embeddings.index_put(
                    (tok_idx_tensor,), local_emb.to(device)
                )

        e_context_raw = token_embeddings

        # ==== 2. Compositional Path (LEAF / PATH) ====
        zero = torch.zeros((L, self.cfg.d_ast), device=device)
        e_leaf_raw = zero
        e_path_raw = zero

        if self.cfg.use_compositional_path:

            def _paths_to_texts(use_type_paths: bool) -> List[str]:
                texts: List[str] = []
                for _, gid in tokens:
                    entry = (type_paths if use_type_paths else value_paths).get(gid)
                    if entry and entry.get("paths"):
                        texts.append(" ".join(str(x) for x in entry["paths"]))
                    else:
                        texts.append("")
                return texts

            def _dedup_embed(texts: List[str]) -> torch.Tensor:
                out = token_embeddings.new_zeros((L, self.cfg.d_ast))
                if not any(texts):
                    return out
                uniq: Dict[str, int] = {}
                uniq_texts: List[str] = []
                for t in texts:
                    if t not in uniq:
                        uniq[t] = len(uniq_texts)
                        uniq_texts.append(t)

                non_empty_map: Dict[int, int] = {}
                uniq_non_empty: List[str] = []
                for u, t in enumerate(uniq_texts):
                    if t != "":
                        non_empty_map[u] = len(uniq_non_empty)
                        uniq_non_empty.append(t)

                if uniq_non_empty:
                    emb_non_empty = self.path_text_embedder(uniq_non_empty)
                else:
                    emb_non_empty = token_embeddings.new_zeros((0, self.cfg.d_ast))

                for i, t in enumerate(texts):
                    u = uniq[t]
                    if t == "":
                        continue
                    out[i] = emb_non_empty[non_empty_map[u]]
                return out

            e_leaf_raw = _dedup_embed(_paths_to_texts(use_type_paths=True))
            e_path_raw = _dedup_embed(_paths_to_texts(use_type_paths=False))
        
        # ==== Affine + Gating ====
        e_context = self.g_context * self.aff_context(e_context_raw)
        e_leaf = self.g_leaf * self.aff_leaf(e_leaf_raw) if self.cfg.use_compositional_path else zero
        e_path = self.g_path * self.aff_path(e_path_raw) if self.cfg.use_compositional_path else zero

        # ==== 3. Symbolic Features ====
        if self.cfg.use_symbolic_feature:
            depths, node_type_ids = [], []
            for i, (tok, gid) in enumerate(tokens):
                if tok in {"(", ")"} or mask_flags_list[i] == 1:
                    depths.append(0)
                    node_type_ids.append(self.ntype2id["NLToken"])
                    continue

                l_entry = type_paths.get(gid)
                p_entry = value_paths.get(gid)
                d = (
                    max(
                        (
                            len(l_entry["paths"])
                            if (l_entry and l_entry.get("paths"))
                            else 1
                        ),
                        (
                            len(p_entry["paths"])
                            if (p_entry and p_entry.get("paths"))
                            else 1
                        ),
                    )
                    - 1
                )
                depths.append(min(d, self.cfg.max_depth))

                ntype = None
                if l_entry and l_entry.get("paths"):
                    cand = l_entry["paths"][-1]
                    if cand in self.ntype2id:
                        ntype = cand
                node_type_ids.append(self.ntype2id.get(ntype, self.ntype2id["NLToken"]))

            depths = torch.tensor(depths, device=device, dtype=torch.long)
            node_type_ids = torch.tensor(node_type_ids, device=device, dtype=torch.long)

            e_pos = self.pos_emb(pos_ids)
            e_depth = self.depth_emb(depths)
            e_ntype = self.node_type_emb(node_type_ids)
            e_seg = self.seg_emb(seg_ids)
            e_modal = self.modality_emb(mod_ids)
        else:
            e_pos = e_depth = e_ntype = e_seg = e_modal = zero

        # ==== Apply Masks and Merge (H0) ====
        # Apply semantic mask (brackets) & mask_flags to content-bearing channels only
        sem = semantic_mask.unsqueeze(-1).float() * (
            1 - mask_flags.unsqueeze(-1).float()
        )
        e_context = e_context * sem
        e_leaf = e_leaf * sem
        e_path = e_path * sem
        e_depth = e_depth * sem
        e_ntype = e_ntype * sem
        e_pos = e_pos * sem
        e_seg = e_seg * sem
        e_modal = e_modal * sem

        # Directly add to form H(0) output after masking all symbolic/content channels.
        h0 = e_leaf + e_path + e_context + e_depth + e_ntype + e_pos + e_seg + e_modal
        h0 = self.ln(self.dropout(h0))

        meta = {
            "positions": pos_ids,
            "depths": (
                depths
                if self.cfg.use_symbolic_feature
                else torch.zeros(L, dtype=torch.long, device=device)
            ),
            "node_type_ids": (
                node_type_ids
                if self.cfg.use_symbolic_feature
                else torch.zeros(L, dtype=torch.long, device=device)
            ),
            "seg_ids": seg_ids,
            "modality_ids": mod_ids,
            "mask_flags": mask_flags,
            "semantic_mask": semantic_mask,
        }

        return {
            "h0": h0,  # SANE Output H(0) [L, D]
            "semantic_mask": semantic_mask,  # Kept at top-level for compatibility
            "meta": meta,
        }
