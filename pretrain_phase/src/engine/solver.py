from src.engine.trainer import Trainer
from src.data.dataset import JepaDataset, jepa_collate_fn
from torch.utils.data import DataLoader
from src.utils.checkpoint import load_t5_model, load_tokenizer
from src.models.factory import ModelFactory
from src.utils.io_utils import make_save_dir
import torch


class Solver:
    def __init__(self, paths, cfg):
        self.paths = paths
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model_dir = make_save_dir(paths.model_dir)

        # data
        train_dataset = JepaDataset(paths, mode="train")
        val_dataset = JepaDataset(paths, mode="val")

        nw = int(getattr(cfg, "num_workers", 8))
        dl_kwargs = {
            "num_workers": nw,
            "pin_memory": True,
        }
        if nw > 0:
            dl_kwargs["persistent_workers"] = bool(
                getattr(cfg, "persistent_workers", True)
            )
            dl_kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 2))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: jepa_collate_fn(b, mask_ratio=cfg.mask_ratio),
            **dl_kwargs,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: jepa_collate_fn(b, mask_ratio=cfg.mask_ratio),
            **dl_kwargs,
        )

        # tokenizer & T5
        self.tokenizer = load_tokenizer(cfg.model_name)
        self.t5_model = load_t5_model(cfg.model_name)

    def train(self):
        # build model
        jepa = ModelFactory.build_jepa(self.cfg, self.tokenizer, self.t5_model)
        # run trainer
        trainer = Trainer(
            self.cfg, self.paths, jepa, self.train_loader, self.val_loader
        )
        trainer.train()
