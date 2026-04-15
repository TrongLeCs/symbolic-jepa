from typing import Any, Dict, List

import wandb


def maybe_log_bias_contrib(jepa, items_full: List[Dict[str, Any]], use_wandb: bool):
    """
    Optionally calculate and log the contribution of different structural bias channels to Weights & Biases.
    This helps in analyzing how much each structural feature (like CSL, LG, NLB) affects the attention mechanism.
    """
    try:
        prev_flag = getattr(jepa.context, "export_bias", False)
        jepa.context.export_bias = True
        pack = jepa.context.forward_batch(items_full)
        R = pack.get("raw_bias", None)  # [B, L, L, C]
        jepa.context.export_bias = prev_flag
        if R is None:
            return

        gamma_c = jepa.context.sgat.gamma_c.detach()  # [C]
        # Calculate the new contribution level: the magnitude of R multiplied by the gamma_c weight of the corresponding channel
        contrib = (R * gamma_c.view(1, 1, 1, -1)).abs().mean(dim=(0, 1, 2))  # [C]

        if use_wandb:
            log_data = {
                f"anlz/bias_contrib[c{c}]": float(contrib[c].item())
                for c in range(contrib.numel())
            }
            wandb.log(log_data, commit=False)
    except Exception:
        pass
