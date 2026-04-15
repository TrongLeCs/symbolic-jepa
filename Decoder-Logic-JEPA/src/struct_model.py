# =============================
# src/struct_model.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import wandb


class T5WithStructHeads(T5ForConditionalGeneration):
    """
    Encoder receives NL; decoder generates FOL.
    Adding 2 auxiliary heads:
      - DFG adjacency (binary, pairwise)
      - AST path (classified by depth & node type)
    """

    config_class = T5Config  # helps HuggingFace identify the config type

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

        # Read hyperparams from config (if saved in config.json) or from kwargs (defaults)
        self.alpha_dfg = getattr(config, "alpha_dfg", kwargs.get("alpha_dfg", 1.0))
        self.alpha_ast = getattr(config, "alpha_ast", kwargs.get("alpha_ast", 1.0))
        self.enable_dfg = getattr(config, "enable_dfg", kwargs.get("enable_dfg", True))
        self.enable_ast = getattr(config, "enable_ast", kwargs.get("enable_ast", True))
        self.num_node_types = getattr(
            config, "num_node_types", kwargs.get("num_node_types", 128)
        )
        self.max_ast_depth = getattr(
            config, "max_ast_depth", kwargs.get("max_ast_depth", 8)
        )
        self.dfg_bits = getattr(config, "dfg_bits", kwargs.get("dfg_bits", 16))
        self.ast_path_bits = getattr(
            config, "ast_path_bits", kwargs.get("ast_path_bits", 128)
        )
        self.wandb_run_name = kwargs.get("wandb_run_name", None)
        print(f"wandb_run_name {self.wandb_run_name}")

        # Initialize heads
        factor = getattr(self.config, "initializer_factor", 1.0)
        d = self.config.d_model

        self.proj_dfg = nn.Linear(d, self.dfg_bits, bias=False)
        self.proj_ast = nn.Linear(d, self.ast_path_bits, bias=False)

        self.dfg_weight1 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_weight2 = nn.Linear(self.dfg_bits, 32, bias=False)
        self.dfg_b1 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b2 = nn.Linear(self.dfg_bits, 1, bias=False)
        self.dfg_b3 = nn.Parameter(torch.tensor(0.0))

        for layer in [
            self.proj_dfg,
            self.proj_ast,
            self.dfg_weight1,
            self.dfg_weight2,
            self.dfg_b1,
            self.dfg_b2,
        ]:
            nn.init.normal_(layer.weight, mean=0.0, std=factor)
        nn.init.normal_(self.dfg_b3, mean=0.0, std=factor)

        # self.ast_path_head = nn.Linear(
        #     self.ast_path_bits, self.max_ast_depth * self.num_node_types, bias=False
        # )
        # nn.init.normal_(self.ast_path_head.weight, mean=0.0, std=factor)
        self.ast_path_head = nn.Linear(
        self.ast_path_bits, self.max_ast_depth * self.num_node_types, bias=True)
        nn.init.normal_(self.ast_path_head.weight, mean=0.0, std=factor)
        nn.init.zeros_(self.ast_path_head.bias)  # or initialize according to prior if statistics are available

        # Ensure hidden states are returned for auxiliary heads
        self.config.output_hidden_states = True

    # ==== Helper: initialize from original T5, then copy weights (strict=False) ====
    @classmethod
    def from_t5_pretrained(cls, base_model_name_or_path: str, **kwargs):
        """
        Used for the first training session when initializing from a pretrained T5:
          model = T5WithStructHeads.from_t5_pretrained("t5-base", alpha_dfg=..., ...)
        """
        # 1) Get config from the original model
        config = T5Config.from_pretrained(base_model_name_or_path)

        # Hyperparams can be saved in the config so they are automatically carried over when saving/loading:
        for k in [
            "enable_dfg",
            "enable_ast",
            "alpha_dfg",
            "alpha_ast",
            "num_node_types",
            "max_ast_depth",
            "dfg_bits",
            "ast_path_bits",
        ]:
            if k in kwargs:
                setattr(config, k, kwargs[k])

        # 2) Create our model
        model = cls(config, **kwargs)

        # 3) Load weights from the original T5 (encoder/decoder/lm_head) with strict=False
        base = T5ForConditionalGeneration.from_pretrained(base_model_name_or_path)
        # encoder_state_dict = torch.load(pretrained_encoder_path, map_location="cuda")
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        # (optional) print for debugging:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    @torch.no_grad()
    def forward2(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **gen_kwargs: Any,
    ) -> torch.LongTensor:
        """
        Inference helper: generate exactly like the standard T5, WITHOUT using AST/DFG.
        Used when you only want NL -> FOL without calculating auxiliary losses.

        Args:
            input_ids: (B, L_in)
            attention_mask: (B, L_in) or None
            **gen_kwargs: parameters for .generate(), for example:
                max_length=64, num_beams=5, no_repeat_ngram_size=2, ...

        Returns:
            sequences (LongTensor): (B, L_out) generated token ids.
        """
        # If the user does not provide max_length or max_new_tokens,
        # set a safe default to avoid the "no maximum length is provided" warning.
        if "max_length" not in gen_kwargs and "max_new_tokens" not in gen_kwargs:
            # priority is given to generation_config if available (read from generation_config.json)
            default_max_len = getattr(
                getattr(self, "generation_config", None), "max_length", None
            )
            gen_kwargs["max_length"] = (
                default_max_len if default_max_len is not None else 128
            )

        sequences = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        return sequences

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        dfg_links: Optional[torch.Tensor] = None,
        ast_paths: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        # Remove keys that may cause conflicts when super().forward is called by generate()
        kwargs.pop("num_items_in_batch", None)

        if labels is None:
            return_dict = kwargs.pop("return_dict", True)
            output_hidden_states = kwargs.pop("output_hidden_states", True)

            # Call standard T5
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # None during inference
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            # Inference: do not add auxiliary loss, return original outputs for generate to use
            if labels is None and dfg_links is None and ast_paths is None:
                return outputs

        # Check if decoder_input_ids exists in kwargs
        if "decoder_input_ids" not in kwargs:
            raise ValueError("decoder_input_ids must be provided in kwargs.")

        # Get decoder_input_ids from kwargs
        decoder_input_ids = kwargs.pop("decoder_input_ids")

        assert not torch.isnan(input_ids).any(), "input_ids contains NaN"
        assert not torch.isnan(labels).any(), "labels contains NaN"
        assert not torch.isnan(
            decoder_input_ids
        ).any(), "decoder_input_ids contains NaN"
        assert not torch.isnan(attention_mask).any(), "attention_mask contains NaN"

        # Encoder takes ONLY NL (input_ids/attention_mask)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # --------- AUXILIARY LOSS CALCULATION DURING TRAINING ----------
        assert not torch.isnan(outputs.loss).any(), "Loss contains NaN!"
        total_loss = outputs.loss
        lm_loss = outputs.loss

        dec_h = outputs.decoder_hidden_states[-1]  # (B, L_out, d)
        # align to predict token t using hidden state at t-1
        seq = dec_h[:, :-1, :]  # (B, L-1, d)
        B, Lm1, _ = seq.shape

        dfg_loss = None
        if self.enable_dfg and dfg_links is not None and self.alpha_dfg != 0:

            # z = seq[:, :, : self.dfg_bits]
            # h1 = self.dfg_weight1(z)
            # h2 = self.dfg_weight2(z)
            # logits_dfg = (
            #     torch.bmm(h1, h2.transpose(1, 2))
            #     + self.dfg_b1(z).squeeze(-1)[:, :, None]
            #     + self.dfg_b2(z).squeeze(-1)[:, None, :]
            #     + self.dfg_b3
            # )

            # === DFG logits (scale stability) ===
            # Get DFG features directly from the decoder hidden state (without using a projection head)
            z = seq[:, :, : self.dfg_bits]  # (B, Lm1, dfg_bits)
            
            # 1) Normalize input features z to reduce scale mismatch between batches/steps.
            #    LayerNorm helps maintain a more stable distribution compared to raw hidden states.
            z_norm = F.layer_norm(z, (z.size(-1),))  # (B, Lm1, dfg_bits)
            
            # 2) Two linear projection branches map to a small space (32) and then L2-normalize.
            #    -> uses cosine similarity, so values are always within [-1, 1]
            h1 = self.dfg_weight1(z_norm)                         # (B, Lm1, 32)
            h2 = self.dfg_weight2(z_norm)                         # (B, Lm1, 32)
            h1 = F.normalize(h1, p=2, dim=-1, eps=1e-6)           # (B, Lm1, 32), ||h1||_2=1
            h2 = F.normalize(h2, p=2, dim=-1, eps=1e-6)           # (B, Lm1, 32), ||h2||_2=1
            
            # 3) Cosine scores (no need to divide by sqrt(dim) since it's already L2-normalized)
            sim = torch.bmm(h1, h2.transpose(1, 2))               # (B, Lm1, Lm1), trong [-1, 1]
            
            # 4) Row/column bias from z_norm to allow the model to learn position-based "activation levels"
            b1 = self.dfg_b1(z_norm).squeeze(-1)                  # (B, Lm1)
            b2 = self.dfg_b2(z_norm).squeeze(-1)                  # (B, Lm1)
            
            # 5) Final logit aggregation (stable, no explosion)
            logits_dfg = sim + b1[:, :, None] + b2[:, None, :] + self.dfg_b3

            if dfg_links.dim() != 3:
                raise ValueError(f"dfg_links.dim()={dfg_links.dim()} != 3")
            B_, L1, L2 = dfg_links.shape
            if B_ != B or L1 != Lm1 or L2 != Lm1:
                raise ValueError(
                    f"DFG shape {dfg_links.shape} != (B={B}, Lm1={Lm1}, Lm1={Lm1})"
                )

            tgt = dfg_links.to(logits_dfg.device).float()  # {0,1,-1}
            pos = tgt == 1
            neg = tgt == 0

            tgt_clamped = torch.clamp(tgt, 0.0, 1.0)

            loss_mat = F.binary_cross_entropy_with_logits(
                logits_dfg, tgt_clamped, reduction="none"
            )

            # average each class separately, then add 1/2 for each side
            pos_sum = (loss_mat * pos).sum()
            neg_sum = (loss_mat * neg).sum()
            pos_cnt = pos.sum().clamp(min=1)
            neg_cnt = neg.sum().clamp(min=1)

            dfg_loss = 0.5 * (pos_sum / pos_cnt) + 0.5 * (neg_sum / neg_cnt)
            total_loss = total_loss + self.alpha_dfg * dfg_loss

        # ===== AST loss =====
        ast_loss = None
        # if ast_paths is not None:
        #     # seq: (B, Lm1, d) đã lấy từ dec_h[:, :-1, :]
        #     # z = self.proj_ast(seq)  # (B, Lm1, ast_path_bits)
        #     z = seq[:, :, self.dfg_bits : self.dfg_bits + self.ast_path_bits]
        #     D = self.max_ast_depth
        #     T = self.num_node_types

        #     logits_ast = self.ast_path_head(z)  # (B, Lm1, D*T)
        #     logits_ast = logits_ast.view(B, Lm1, D, T)  # (B, Lm1, D, T)

        #     # ---- Strict shape check: ast_paths phải là (B, Lm1, D) KHÔNG BOS/EOS, pad = -1
        #     if ast_paths.dim() != 3:
        #         raise ValueError(
        #             f"ast_paths.dim()={ast_paths.dim()} != 3 (expect B,Lm1,D)"
        #         )
        #     B_, L1, D_ = ast_paths.shape
        #     if B_ != B or L1 != Lm1 or D_ != D:
        #         raise ValueError(
        #             f"AST shape {tuple(ast_paths.shape)} != (B={B}, Lm1={Lm1}, D={D}). "
        #             "Đảm bảo collator đã pad/crop đúng và không thêm BOS/EOS."
        #         )

        #     tgt_paths = ast_paths.to(
        #         logits_ast.device
        #     ).long()  # (B, Lm1, D), values in {-1, 0..T-1}

        #     # CrossEntropy with ignore_index=-1 to skip padding
        #     ce = nn.CrossEntropyLoss(ignore_index=-1)
        #     ast_loss = ce(
        #         logits_ast.reshape(-1, T),  # ((B*Lm1*D), T)
        #         tgt_paths.reshape(-1),  # ((B*Lm1*D),)
        #     )
        #     total_loss = total_loss + self.alpha_ast * ast_loss

        if self.enable_ast and ast_paths is not None and self.alpha_ast != 0:
            # seq: (B, Lm1, d) taken from dec_h[:, :-1, :]
            # Directly use LM hidden states sliced for AST (without using projection)
            z = seq[:, :, self.dfg_bits : self.dfg_bits + self.ast_path_bits]  # (B, Lm1, ast_bits)
        
            D = self.max_ast_depth
            T = self.num_node_types
        
            # 1) Stabilize input distribution
            z = F.layer_norm(z, (z.size(-1),))                      # (B, Lm1, ast_bits)
            z = F.dropout(z, p=0.1, training=self.training)         # small amount to reduce over-confidence
        
            # 2) Create logits and scale to a safe range
            logits_ast = self.ast_path_head(z)                      # (B, Lm1, D*T)
            logits_ast = logits_ast / (self.ast_path_bits ** 0.5)   # reduce scale
            logits_ast = logits_ast.view(B, Lm1, D, T)              # (B, Lm1, D, T)
        
            # ---- Strict shape check: ast_paths must be (B, Lm1, D) WITHOUT BOS/EOS, pad = -1
            if ast_paths.dim() != 3:
                raise ValueError(f"ast_paths.dim()={ast_paths.dim()} != 3 (expect B,Lm1,D)")
            B_, L1, D_ = ast_paths.shape
            if B_ != B or L1 != Lm1 or D_ != D:
                raise ValueError(
                    f"AST shape {tuple(ast_paths.shape)} != (B={B}, Lm1={Lm1}, D={D}). "
                # Ensure the collator has correctly padded/cropped and does not add BOS/EOS.
                )
        
            tgt_paths = ast_paths.to(logits_ast.device).long()  # (B, Lm1, D), giá trị ∈ {-1, 0..T-1}
        
            # 3) CE with ignore pad; optional slight label smoothing (e.g., 0.05)
            ce = nn.CrossEntropyLoss(ignore_index=-1)  # or nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
            ast_loss = ce(
                logits_ast.reshape(-1, T),    # ((B*Lm1*D), T)
                tgt_paths.reshape(-1),        # ((B*Lm1*D),)
            )
        
            total_loss = total_loss + self.alpha_ast * ast_loss


        # Log to WandB
        if self.wandb_run_name:
            wandb.log(
                {
                    "loss": total_loss.item(),
                    "lm_loss": lm_loss.item(),
                    "dfg_loss": dfg_loss.item() if dfg_loss is not None else None,
                    "ast_loss": ast_loss.item() if ast_loss is not None else None,
                }
            )

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "lm_loss": lm_loss,
            "dfg_loss": dfg_loss,
            "ast_loss": ast_loss,
        }
