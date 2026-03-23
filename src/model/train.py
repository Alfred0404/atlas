import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .config import ModelConfig
from ..utils.logging import setup_logging

logger = setup_logging()


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: ModelConfig,
        val_loader: DataLoader = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Linear warmup + cosine annealing
        total_steps = config.max_epochs * len(train_loader)

        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(
                1, total_steps - config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

        # History for plots
        self.history = {
            "step_loss": [],
            "step_lr": [],
            "epoch_train_loss": [],
            "epoch_val_loss": [],
        }
        self.start_time = None

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        self.start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            self.history["epoch_train_loss"].append(train_loss)
            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} — train loss: {train_loss:.4f}"
            )

            if self.val_loader is not None:
                val_loss = self._validate()
                self.history["epoch_val_loss"].append(val_loss)
                logger.info(f"Epoch {epoch} — val loss: {val_loss:.4f}")
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=is_best)
            else:
                self._save_checkpoint(epoch, train_loss)

        self._print_summary()
        self._plot_curves()

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x, mask_logits=True)  # (B, S, V)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            self.global_step += 1

            lr = self.scheduler.get_last_lr()[0]
            self.history["step_loss"].append(step_loss)
            self.history["step_lr"].append(lr)

            if self.global_step % self.config.log_interval == 0:
                logger.info(
                    f"  step {self.global_step} — loss: {step_loss:.4f}, lr: {lr:.2e}"
                )

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x, mask_logits=True)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "global_step": self.global_step,
            "config": self.config,
        }
        path = os.path.join(self.config.checkpoint_dir, "atlas_transformer.pt")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir, "atlas_transformer_best.pt"
            )
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path}")

    def _print_summary(self) -> None:
        """Print training summary stats."""
        elapsed = time.time() - self.start_time
        minutes = elapsed / 60
        train_losses = self.history["epoch_train_loss"]
        val_losses = self.history["epoch_val_loss"]

        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Duration: {minutes:.1f} min")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Epochs: {len(train_losses)}")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Best train loss:  {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1})")
        if val_losses:
            logger.info(f"Final val loss:   {val_losses[-1]:.4f}")
            logger.info(f"Best val loss:    {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
        logger.info("=" * 50)

    def _plot_curves(self) -> None:
        """Save training curves to checkpoint directory."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Step loss (smoothed)
        step_losses = self.history["step_loss"]
        axes[0].plot(step_losses, alpha=0.3, color="blue", label="raw")
        # Smoothed with rolling average
        window = max(1, len(step_losses) // 50)
        if len(step_losses) > window:
            smoothed = [
                sum(step_losses[max(0, i - window):i + 1]) / min(i + 1, window + 1)
                for i in range(len(step_losses))
            ]
            axes[0].plot(smoothed, color="blue", label="smoothed")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Step Loss")
        axes[0].legend()

        # Epoch loss
        epochs = range(1, len(self.history["epoch_train_loss"]) + 1)
        axes[1].plot(epochs, self.history["epoch_train_loss"], marker="o", markersize=2, label="train")
        if self.history["epoch_val_loss"]:
            axes[1].plot(epochs, self.history["epoch_val_loss"], marker="o", markersize=2, label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Epoch Loss")
        axes[1].legend()

        # Learning rate
        axes[2].plot(self.history["step_lr"])
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")

        plt.tight_layout()
        path = os.path.join(self.config.checkpoint_dir, "training_curves.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Training curves saved to {path}")

    @staticmethod
    def load_checkpoint(
        path: str, model: nn.Module, device: str = "cuda"
    ) -> dict:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
