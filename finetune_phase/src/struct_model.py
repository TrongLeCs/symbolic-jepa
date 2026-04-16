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
      - LDP adjacency (binary, pairwise)
      - CPP path (classified by depth & node type)
    """

    config_class = T5Config  # helps HuggingFace identify the config type

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

        # Read hyperparams from config (if saved in config.json) or from kwargs (defaults)
        self.alpha_ldp = getattr(config, "alpha_ldp", kwargs.get("alpha_ldp", 1.0))
        self.alpha_cpp = getattr(config, "alpha_cpp", kwargs.get("alpha_cpp", 1.0))
        self.enable_ldp = getattr(config, "enable_ldp", kwargs.get("enable_ldp", True))
        self.enable_cpp = getattr(config, "enable_cpp", kwargs.get("enable_cpp", True))
        self.num_node_types = getattr(
            config, "num_node_types", kwargs.get("num_node_types", 128)
        )
        self.max_cpp_depth = getattr(
            config, "max_cpp_depth", kwargs.get("max_cpp_depth", 8)
        )
        self.ldp_bits = getattr(config, "ldp_bits", kwargs.get("ldp_bits", 16))
        self.cpp_path_bits = getattr(
            config, "cpp_path_bits", kwargs.get("cpp_path_bits", 128)
        )
        self.wandb_run_name = kwargs.get("wandb_run_name", None)
        print(f"wandb_run_name {self.wandb_run_name}")

        # Initialize heads
        factor = getattr(self.config, "initializer_factor", 1.0)
        d = self.config.d_model

        self.proj_ldp = nn.Linear(d, self.ldp_bits, bias=False)
        self.proj_cpp = nn.Linear(d, self.cpp_path_bits, bias=False)

        self.ldp_weight1 = nn.Linear(self.ldp_bits, 32, bias=False)
        self.ldp_weight2 = nn.Linear(self.ldp_bits, 32, bias=False)
        self.ldp_b1 = nn.Linear(self.ldp_bits, 1, bias=False)
        self.ldp_b2 = nn.Linear(self.ldp_bits, 1, bias=False)
        self.ldp_b3 = nn.Parameter(torch.tensor(0.0))

        for layer in [
            self.proj_ldp,
            self.proj_cpp,
            self.ldp_weight1,
            self.ldp_weight2,
            self.ldp_b1,
            self.ldp_b2,
        ]:
            nn.init.normal_(layer.weight, mean=0.0, std=factor)
        nn.init.normal_(self.ldp_b3, mean=0.0, std=factor)

        # self.ast_path_head = nn.Linear(
        #     self.ast_path_bits, self.max_ast_depth * self.num_node_types, bias=False
        # )
        # nn.init.normal_(self.ast_path_head.weight, mean=0.0, std=factor)
        self.cpp_path_head = nn.Linear(
        self.cpp_path_bits, self.max_cpp_depth * self.num_node_types, bias=True)
        nn.init.normal_(self.cpp_path_head.weight, mean=0.0, std=factor)
        nn.init.zeros_(self.cpp_path_head.bias)  # or initialize according to prior if statistics are available

        # Ensure hidden states are returned for auxiliary heads
        self.config.output_hidden_states = True

    # ==== Helper: initialize from original T5, then copy weights (strict=False) ====
    @classmethod
    def from_t5_pretrained(cls, base_model_name_or_path: str, **kwargs):
        """
        Used for the first training session when initializing from a pretrained T5:
          model = T5WithStructHeads.from_t5_pretrained("t5-base", alpha_ldp=..., ...)
        """
        # 1) Get config from the original model
        config = T5Config.from_pretrained(base_model_name_or_path)

        # Hyperparams can be saved in the config so they are automatically carried over when saving/loading:
        for k in [
            "enable_ldp",
            "enable_cpp",
            "alpha_ldp",
            "alpha_cpp",
            "num_node_types",
            "max_cpp_depth",
            "ldp_bits",
            "cpp_path_bits",
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
        Inference helper: generate exactly like the standard T5, WITHOUT using CPP/LDP.
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
        ldp_links: Optional[torch.Tensor] = None,
        cpp_paths: Optional[torch.Tensor] = None,
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
            if labels is None and ldp_links is None and cpp_paths is None:
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

        ldp_loss = None
        if self.enable_ldp and ldp_links is not None and self.alpha_ldp != 0:

            # z = seq[:, :, : self.ldp_bits]
            # h1 = self.ldp_weight1(z)
            # h2 = self.ldp_weight2(z)
            # logits_ldp = (
            #     torch.bmm(h1, h2.transpose(1, 2))
            #     + self.ldp_b1(z).squeeze(-1)[:, :, None]
            #     + self.ldp_b2(z).squeeze(-1)[:, None, :]
            #     + self.ldp_b3
            # )

            # === LDP logits (scale stability) ===
            # Get LDP features directly from the decoder hidden state (without using a projection head)
            z = seq[:, :, : self.ldp_bits]  # (B, Lm1, ldp_bits)
            
            # 1) Normalize input features z to reduce scale mismatch between batches/steps.
            #    LayerNorm helps maintain a more stable distribution compared to raw hidden states.
            z_norm = F.layer_norm(z, (z.size(-1),))  # (B, Lm1, ldp_bits)
            
            # 2) Two linear projection branches map to a small space (32) and then L2-normalize.
            #    -> uses cosine similarity, so values are always within [-1, 1]
            h1 = self.ldp_weight1(z_norm)                         # (B, Lm1, 32)
            h2 = self.ldp_weight2(z_norm)                         # (B, Lm1, 32)
            h1 = F.normalize(h1, p=2, dim=-1, eps=1e-6)           # (B, Lm1, 32), ||h1||_2=1
            h2 = F.normalize(h2, p=2, dim=-1, eps=1e-6)           # (B, Lm1, 32), ||h2||_2=1
            
            # 3) Cosine scores (no need to divide by sqrt(dim) since it's already L2-normalized)
            sim = torch.bmm(h1, h2.transpose(1, 2))               # (B, Lm1, Lm1), trong [-1, 1]
            
            # 4) Row/column bias from z_norm to allow the model to learn position-based "activation levels"
            b1 = self.ldp_b1(z_norm).squeeze(-1)                  # (B, Lm1)
            b2 = self.ldp_b2(z_norm).squeeze(-1)                  # (B, Lm1)
            
            # 5) Final logit aggregation (stable, no explosion)
            logits_ldp = sim + b1[:, :, None] + b2[:, None, :] + self.ldp_b3

            if ldp_links.dim() != 3:
                raise ValueError(f"ldp_links.dim()={ldp_links.dim()} != 3")
            B_, L1, L2 = ldp_links.shape
            if B_ != B or L1 != Lm1 or L2 != Lm1:
                raise ValueError(
                    f"LDP shape {ldp_links.shape} != (B={B}, Lm1={Lm1}, Lm1={Lm1})"
                )

            tgt = ldp_links.to(logits_ldp.device).float()  # {0,1,-1}
            pos = tgt == 1
            neg = tgt == 0

            tgt_clamped = torch.clamp(tgt, 0.0, 1.0)

            loss_mat = F.binary_cross_entropy_with_logits(
                logits_ldp, tgt_clamped, reduction="none"
            )

            # average each class separately, then add 1/2 for each side
            pos_sum = (loss_mat * pos).sum()
            neg_sum = (loss_mat * neg).sum()
            pos_cnt = pos.sum().clamp(min=1)
            neg_cnt = neg.sum().clamp(min=1)

            ldp_loss = 0.5 * (pos_sum / pos_cnt) + 0.5 * (neg_sum / neg_cnt)
            total_loss = total_loss + self.alpha_ldp * ldp_loss

        # ===== CPP loss =====
        cpp_loss = None
        if self.enable_cpp and cpp_paths is not None and self.alpha_cpp != 0:
            # seq: (B, Lm1, d) taken from dec_h[:, :-1, :]
            # Directly use LM hidden states sliced for CPP (without using projection)
            z = seq[:, :, self.ldp_bits : self.ldp_bits + self.cpp_path_bits]  # (B, Lm1, cpp_bits)
        
            D = self.max_cpp_depth
            T = self.num_node_types
        
            # 1) Stabilize input distribution
            z = F.layer_norm(z, (z.size(-1),))                      # (B, Lm1, cpp_bits)
            z = F.dropout(z, p=0.1, training=self.training)         # small amount to reduce over-confidence
        
            # 2) Create logits and scale to a safe range
            logits_cpp = self.cpp_path_head(z)                      # (B, Lm1, D*T)
            logits_cpp = logits_cpp / (self.cpp_path_bits ** 0.5)   # reduce scale
            logits_cpp = logits_cpp.view(B, Lm1, D, T)              # (B, Lm1, D, T)
        
            # ---- Strict shape check: cpp_paths must be (B, Lm1, D) WITHOUT BOS/EOS, pad = -1
            if cpp_paths.dim() != 3:
                raise ValueError(f"cpp_paths.dim()={cpp_paths.dim()} != 3 (expect B,Lm1,D)")
            B_, L1, D_ = cpp_paths.shape
            if B_ != B or L1 != Lm1 or D_ != D:
                raise ValueError(
                    f"CPP shape {tuple(cpp_paths.shape)} != (B={B}, Lm1={Lm1}, D={D}). "
                )
        
            tgt_paths = cpp_paths.to(logits_cpp.device).long()  # (B, Lm1, D), giá trị ∈ {-1, 0..T-1}
        
            # 3) CE with ignore pad
            ce = nn.CrossEntropyLoss(ignore_index=-1)
            cpp_loss = ce(
                logits_cpp.reshape(-1, T),    # ((B*Lm1*D), T)
                tgt_paths.reshape(-1),        # ((B*Lm1*D),)
            )
        
            total_loss = total_loss + self.alpha_cpp * cpp_loss


        # Log to WandB
        if self.wandb_run_name:
            wandb.log(
                {
                    "loss": total_loss.item(),
                    "lm_loss": lm_loss.item(),
                    "ldp_loss": ldp_loss.item() if ldp_loss is not None else None,
                    "cpp_loss": cpp_loss.item() if cpp_loss is not None else None,
                }
            )

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "lm_loss": lm_loss,
            "ldp_loss": ldp_loss,
            "cpp_loss": cpp_loss,
        }
