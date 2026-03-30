"""Training loop for the GraphTransformer model.

Usage (see graph_train_model.py at project root):
    trainer = GraphTrainer(cfg, model, train_dataset, val_dataset)
    trainer.train()
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from src.data.graph_dataset import AssemblyStepDataset, collate_fn
from src.model.graph_config import GraphModelConfig
from src.model.graph_transformer import GraphTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def _cosine_warmup_schedule(warmup_steps: int, total_steps: int):
    def _fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GraphTrainer:
    """Manages training and validation of a ``GraphTransformer``.

    Parameters
    ----------
    cfg : GraphModelConfig
    model : GraphTransformer
    train_dataset / val_dataset : AssemblyStepDataset
    device : torch.device or str
    loss_weights : optional per-head loss weights (default all 1.0)
    """

    def __init__(
        self,
        cfg: GraphModelConfig,
        model: GraphTransformer,
        train_dataset: AssemblyStepDataset,
        val_dataset: AssemblyStepDataset,
        device: torch.device | str = "cpu",
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.device = torch.device(device)
        self.loss_weights = loss_weights or {}

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=str(device).startswith("cuda"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=str(device).startswith("cuda"),
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        total_steps = len(self.train_loader) * cfg.max_epochs
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_cosine_warmup_schedule(cfg.warmup_steps, total_steps),
        )

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._global_step = 0
        self._best_val_loss = float("inf")

        logger.info(
            "Trainer ready | device=%s | batch_size=%d | train_batches=%d | val_batches=%d",
            self.device,
            self.cfg.batch_size,
            len(self.train_loader),
            len(self.val_loader),
        )

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Training started for %d epochs", self.cfg.max_epochs)
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss, train_heads = self._run_epoch(train=True)
            val_loss, val_heads = self._run_epoch(train=False)

            logger.info(
                "epoch %d/%d  train=%.4f  val=%.4f  lr=%.2e",
                epoch,
                self.cfg.max_epochs,
                train_loss,
                val_loss,
                self.scheduler.get_last_lr()[0],
            )
            logger.info(
                "  train heads — %s",
                "  ".join(f"{k}:{v:.3f}" for k, v in train_heads.items()),
            )
            logger.info(
                "  val   heads — %s",
                "  ".join(f"{k}:{v:.3f}" for k, v in val_heads.items()),
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                logger.info("  → new best val loss %.4f, checkpoint saved", val_loss)

            # Always keep a rolling checkpoint every epoch for crash recovery.
            self.save_checkpoint("latest.pt")
            logger.info("  → latest checkpoint saved")

            if epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt")

    # ------------------------------------------------------------------

    def _run_epoch(self, train: bool) -> tuple[float, dict[str, float]]:
        """Run one full pass over the dataset.

        Returns (mean_total_loss, mean_per_head_losses).
        """
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader

        total_loss = 0.0
        head_sums: dict[str, float] = {}
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = _to_device(batch, self.device)

                out = self.model(
                    batch["graph"],
                    batch["open_ports"],
                    batch["open_port_mask"],
                )

                targets = dict(
                    port=batch["target_port_idx"],
                    part_id=batch["target_part_id"],
                    color=batch["target_color"],
                    self_port=batch["target_self_port"],
                    rot_steps=batch["target_rot_steps"],
                )
                loss, per_head = GraphTransformer.compute_loss(
                    out, targets, self.loss_weights
                )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self._global_step += 1

                    if self._global_step % self.cfg.log_interval == 0:
                        logger.info(
                            "step %d  loss=%.4f  lr=%.2e",
                            self._global_step,
                            loss.item(),
                            self.scheduler.get_last_lr()[0],
                        )

                total_loss += loss.item()
                for k, v in per_head.items():
                    head_sums[k] = head_sums.get(k, 0.0) + float(v)
                n_batches += 1

        if n_batches == 0:
            return 0.0, {}

        mean_heads = {k: v / n_batches for k, v in head_sums.items()}
        return total_loss / n_batches, mean_heads

    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / filename
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "global_step": self._global_step,
                "best_val_loss": self._best_val_loss,
                "cfg": self.cfg,
            },
            path,
        )
        logger.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, filename: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / filename
        torch.serialization.add_safe_globals([GraphModelConfig])
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._global_step = ckpt["global_step"]
        self._best_val_loss = ckpt["best_val_loss"]
        logger.info("Loaded checkpoint %s (step %d)", path, self._global_step)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a collated batch to *device*."""
    result = {}
    for k, v in batch.items():
        if k == "graph":
            result[k] = v.to(device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Convenience builder used by graph_train_model.py
# ---------------------------------------------------------------------------


def build_trainer(
    graph_dir: str,
    cfg: GraphModelConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_fraction: float = 0.1,
) -> GraphTrainer:
    """Load dataset, split train/val, build model and trainer.

    Parameters
    ----------
    graph_dir : str
        Directory containing ``.npz`` graph files.
    cfg : GraphModelConfig
    device : str
    val_fraction : float
        Fraction of the dataset to use for validation.
    """
    t0 = time.perf_counter()
    logger.info("Loading graph dataset from %s", graph_dir)
    full_ds = AssemblyStepDataset(graph_dir)
    t1 = time.perf_counter()
    logger.info("Dataset ready in %.1fs | samples=%d", t1 - t0, len(full_ds))

    n_val = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    logger.info("Splitting dataset | train=%d val=%d", n_train, n_val)
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    model = GraphTransformer(cfg)
    logger.info(
        "Model: %d parameters  |  train steps: %d  |  val steps: %d",
        sum(p.numel() for p in model.parameters()),
        n_train,
        n_val,
    )
    logger.info("Build trainer completed in %.1fs", time.perf_counter() - t0)

    return GraphTrainer(cfg, model, train_ds, val_ds, device=device)
