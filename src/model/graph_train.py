"""Training loop for the GraphTransformer model.

Usage (see graph_train_model.py at project root):
    trainer = GraphTrainer(cfg, model, train_dataset, val_dataset)
    trainer.train()
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
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
# Live plot worker (separate process — keeps the window responsive)
# ---------------------------------------------------------------------------


def _plot_worker(conn) -> None:
    """Runs in a child process: reads from a Pipe connection and redraws the loss figure."""
    import matplotlib.pyplot as plt

    train_losses: list[float] = []
    val_losses: list[float] = []
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)
    plt.pause(0.1)

    while True:
        if conn.poll(0.5):
            item = conn.recv()
            if item is None:  # shutdown sentinel
                plt.ioff()
                plt.show(block=True)
                return
            train_losses.append(item[0])
            val_losses.append(item[1])
            epochs = list(range(1, len(train_losses) + 1))
            ax.clear()
            ax.plot(epochs, train_losses, label="train")
            ax.plot(epochs, val_losses, label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_ylim(0, 25)
            ax.legend()
            fig.tight_layout()
            fig.canvas.draw()
        plt.pause(0.1)


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
        live_plot: bool = False,
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
        self._epochs_without_improvement = 0
        self._live_plot = live_plot
        if live_plot:
            self._init_plot()

        logger.info(
            "Trainer ready | device=%s | batch_size=%d | train_batches=%d | val_batches=%d",
            self.device,
            self.cfg.batch_size,
            len(self.train_loader),
            len(self.val_loader),
        )

    # ------------------------------------------------------------------

    def _init_plot(self) -> None:
        try:
            import matplotlib  # noqa: F401 — check availability before spawning
        except ImportError:
            logger.warning("matplotlib not installed — live plot disabled")
            self._live_plot = False
            return
        child_conn, self._plot_conn = mp.Pipe(duplex=False)
        self._plot_proc = mp.Process(
            target=_plot_worker, args=(child_conn,), daemon=True
        )
        self._plot_proc.start()

    def _update_plot(self, train_loss: float, val_loss: float) -> None:
        self._plot_conn.send((train_loss, val_loss))

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

            if self._live_plot:
                self._update_plot(train_loss, val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._epochs_without_improvement = 0
                self.save_checkpoint("best.pt")
                logger.info("  → new best val loss %.4f, checkpoint saved", val_loss)
            else:
                self._epochs_without_improvement += 1

            # Always keep a rolling checkpoint every epoch for crash recovery.
            self.save_checkpoint("latest.pt")
            logger.info("  → latest checkpoint saved")

            if epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt")

            patience = self.cfg.early_stopping_patience
            if patience > 0 and self._epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping: val loss did not improve for %d epochs.", patience
                )
                break

        if self._live_plot:
            self._plot_conn.send(None)  # sentinel: worker shows final plot, blocks until closed
            logger.info("Training complete. Close the plot window to exit.")
            self._plot_proc.join()

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
                    target_port_idx=batch["target_port_idx"] if train else None,
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
                            "step %d  loss=%.4f  lr=%.2e  heads: %s",
                            self._global_step,
                            loss.item(),
                            self.scheduler.get_last_lr()[0],
                            "  ".join(f"{k}:{float(v):.3f}" for k, v in per_head.items()),
                        )

                    interval = self.cfg.checkpoint_interval
                    if interval > 0 and self._global_step % interval == 0:
                        self.save_checkpoint("latest.pt")

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
                "epochs_no_improve": self._epochs_without_improvement,
                "cfg": self.cfg,
            },
            path,
        )
        logger.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, filename: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / filename
        torch.serialization.add_safe_globals([GraphModelConfig])
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        saved_cfg = ckpt.get("cfg")
        if saved_cfg is not None and (
            saved_cfg.n_parts != self.cfg.n_parts
            or saved_cfg.n_colors != self.cfg.n_colors
        ):
            raise ValueError(
                f"Checkpoint vocab mismatch: saved n_parts={saved_cfg.n_parts}, "
                f"n_colors={saved_cfg.n_colors} vs current n_parts={self.cfg.n_parts}, "
                f"n_colors={self.cfg.n_colors}. Delete the checkpoint or pass --no-resume."
            )
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._global_step = ckpt["global_step"]
        self._best_val_loss = ckpt["best_val_loss"]
        self._epochs_without_improvement = ckpt.get("epochs_no_improve", 0)
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
    live_plot: bool = False,
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

    return GraphTrainer(cfg, model, train_ds, val_ds, device=device, live_plot=live_plot)
