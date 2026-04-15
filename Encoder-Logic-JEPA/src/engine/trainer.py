import math, os, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.engine.amp import AmpKit
from src.engine.profiler import maybe_log_bias_contrib
from src.data.linearizer import linearize_sample
from src.data.masking import sample_mask_flags
import wandb


class Trainer:
    """
    Handles the training loop, validation, and checkpointing for the Logic-JEPA model.
    """

    def __init__(self, cfg, paths, model, train_loader, val_loader):
        """
        Initialize the Trainer with configurations, paths, model, and dataloaders.
        """
        self.cfg = cfg
        self.paths = paths
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(cfg.device)

        if cfg.use_wandb:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Logic-JEPA-Pretrain"),
                name=os.environ.get("WANDB_NAME", "Logic-JEPA-Run"),
                config=vars(self.cfg),
            )

        self.best_val_loss = float("inf")
        self.steps_per_epoch = len(self.train_loader)
        # Current EMA momentum (placeholder, will be updated per epoch in on_epoch_start)
        self.ema_m_now = float(getattr(self.cfg, "ema_decay_warm", self.cfg.ema_decay))

    # ---------- Hooks ----------
    def on_epoch_start(self, epoch: int):
        """
        Hook called at the beginning of each epoch.
        Handles the 2-stage fine-tuning schedule for T5 and updates the EMA momentum.
        """
        # ---------- 2-stage T5 finetune ----------
        enable_t5_ft = bool(getattr(self.cfg, "fine_tune_t5", True))
        t5_ft_epoch = int(getattr(self.cfg, "t5_ft_start_epoch", 3))
        do_ft = enable_t5_ft and (epoch >= t5_ft_epoch)
        if hasattr(self.model.context.sane, "set_finetune"):
            self.model.context.sane.set_finetune(do_ft)
        # (Optional) if the predictor has its own T5 and supports set_finetune:
        if hasattr(self.model.predictor, "set_finetune"):
            try:
                self.model.predictor.set_finetune(do_ft)
            except Exception:
                pass

        # ---------- EMA schedule per epoch ----------
        ema_warm = float(
            getattr(self.cfg, "ema_decay_warm", 0.9995)
        )  # Used before unfreezing T5
        ema_after = float(
            getattr(self.cfg, "ema_decay", 0.996)
        )  # Used after unfreezing T5
        # If T5 fine-tune is disabled, always stay in warm EMA regime.
        if not enable_t5_ft:
            self.ema_m_now = ema_warm
        else:
            # Warm-up when epoch < T5 fine-tuning start epoch
            self.ema_m_now = ema_warm if (epoch < t5_ft_epoch) else ema_after

        # ---------- Log gamma_c, b0, and current EMA ----------
        gamma_c, b0 = None, None
        try:
            # Get the weight vector of the 3 structural channels (CSL, LG, NLB)
            gamma_c = self.model.context.sgat.gamma_c.detach().cpu().tolist()
            # Get the global offset value b0
            b0 = float(self.model.context.sgat.b0.detach().item())
        except Exception:
            pass

        if self.cfg.use_wandb:
            log = {
                "train/t5_finetune": float(do_ft),
                "train/ema_decay_now": float(self.ema_m_now),
                "epoch": epoch,
            }
            if b0 is not None:
                log["train/struct_b0"] = b0
            if gamma_c is not None:
                # Log the weight of each individual structural channel
                for i, gc in enumerate(gamma_c):
                    log[f"train/struct_gamma_c[{i}]"] = gc
            wandb.log(log, commit=False)

    # ---------- Main loop ----------
    def train(self):
        """
        Execute the main training loop, including forward passes, backpropagation,
        gradient scaling (AMP), EMA target updates, and logging.
        """
        amp = AmpKit(self.device.type)
        scaler = amp.scaler
        optim = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        grad_clip = self.cfg.grad_clip

        data_iterator = iter(self.train_loader)
        total_train_samples = len(self.train_loader.dataset)
        global_update_step = 0
        running = 0.0

        optim.zero_grad(set_to_none=True)

        target_update_samples = int(self.cfg.update_every_samples)
        if target_update_samples <= 0:
            raise ValueError("update_every_samples must be > 0")

        accumulation_steps = int(getattr(self.cfg, "accumulation_steps", 1))
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be > 0")

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.on_epoch_start(epoch)
            samples_since_update = 0
            t0 = time.time()

            pbar = tqdm(
                total=total_train_samples, desc=f"Epoch {epoch} [train]", unit="sample"
            )
            for step in range(self.steps_per_epoch):
                try:
                    items_full, mask_flags_batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.train_loader)
                    items_full, mask_flags_batch = next(data_iterator)

                bs = len(items_full)

                self.model.train()
                with amp.autocast():
                    out = self.model(
                        items_full=items_full,
                        mask_flags=mask_flags_batch,
                        ema_m=self.ema_m_now,
                    )
                    loss = out["loss"]

                if not torch.isfinite(loss):
                    print(f"[Warn] Non-finite loss at step {step+1}, skip batch")
                    continue

                scaler.scale(loss / accumulation_steps).backward()
                running += float(loss.detach().item())
                samples_since_update += bs

                if samples_since_update >= target_update_samples:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip
                    )
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

                    if hasattr(self.model, "update_target"):
                        self.model.update_target(self.ema_m_now)

                    global_update_step += 1
                    samples_since_update = 0

                pbar.update(bs)
                pbar.set_postfix(
                    {
                        "loss": f"{float(loss.detach().item()):.4f}",
                        "upd": global_update_step,
                    },
                    refresh=False,
                )

                if (step + 1) % self.cfg.log_every == 0:
                    avg = running / self.cfg.log_every
                    if self.cfg.use_wandb:
                        log = {"train/loss": avg, "updates": global_update_step}
                        for k in ("loss_main", "loss_xl", "num_masked"):
                            if k in out:
                                v = out[k]
                                log[f"train/{k}"] = (
                                    float(v)
                                    if isinstance(v, (int, float))
                                    else float(getattr(v, "item", lambda: v)())
                                )
                        log.update(
                            {
                                "cfg/use_compositional_path": float(
                                    self.cfg.sane_cfg.use_compositional_path
                                ),
                                "cfg/use_symbolic_feature": float(
                                    self.cfg.sane_cfg.use_symbolic_feature
                                ),
                                "cfg/fine_tune_t5": float(self.cfg.fine_tune_t5),
                            }
                        )
                        wandb.log(log, step=global_update_step)
                    running = 0.0

            pbar.close()

            # Flush remaining samples
            if samples_since_update > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if hasattr(self.model, "update_target"):
                    self.model.update_target(self.ema_m_now)
                global_update_step += 1

            # Validate (using the current ema_m_now)
            val_loss = self.validate(self.ema_m_now)
            print(
                f"[Epoch {epoch} done in {time.time()-t0:.1f}s] val_loss={val_loss:.6f} | total_updates={global_update_step}"
            )
            if self.cfg.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "epoch": epoch,
                        "updates": global_update_step,
                        "cfg/use_compositional_path": float(
                            self.cfg.sane_cfg.use_compositional_path
                        ),
                        "cfg/use_symbolic_feature": float(
                            self.cfg.sane_cfg.use_symbolic_feature
                        ),
                        "cfg/fine_tune_t5": float(self.cfg.fine_tune_t5),
                    },
                    step=global_update_step,
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_all(
                    self.model, epoch, self.best_val_loss, optim, global_update_step
                )

        print("[Done] Training finished.]")

    @torch.no_grad()
    def validate(self, ema_m: float):
        """
        Evaluate the model on the validation dataset and return the average loss.
        """
        total_loss = 0.0
        n_batches = len(self.val_loader)
        target_probes = 12
        k = max(1, round(n_batches / target_probes))

        self.model.eval()
        pbar = tqdm(total=len(self.val_loader.dataset), desc="Validate", unit="sample")

        for n, (items_full, mask_flags_batch) in enumerate(self.val_loader, start=1):
            bs = len(items_full)

            out = self.model(
                items_full=items_full, mask_flags=mask_flags_batch, ema_m=ema_m
            )
            total_loss += float(out["loss"].item())

            if (n % k) == 1:
                maybe_log_bias_contrib(self.model, items_full, self.cfg.use_wandb)

            pbar.update(bs)
            pbar.set_postfix(
                {"loss": f"{float(out['loss'].item()):.4f}"}, refresh=False
            )
        pbar.close()

        return total_loss / max(n_batches, 1)

    def save_all(self, jepa, epoch, best_loss, optim=None, global_step=None):
        """
        Save the model checkpoints, including encoders, predictor, optimizer state,
        and random number generator states to resume training if necessary.
        """
        ckpt = {
            "context_encoder": jepa.context.state_dict(),
            "target_encoder": jepa.target.state_dict(),
            "predictor": jepa.predictor.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "global_step": global_step,
            "cfg": vars(self.cfg),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        save_path = os.path.join(self.paths.model_dir, "mixed_jepa_best.pt")
        torch.save(ckpt, save_path)
        print(f"[Info] Saved best model -> {save_path} (loss={best_loss:.6f})")

        # Save the target T5 encoder
        t5 = jepa.target.sane.t5
        out_path = os.path.join(self.paths.model_dir, "t5_target_encoder.pth")
        torch.save(t5.state_dict(), out_path)
        print(f"[Info] Saved target T5 encoder -> {out_path}")
