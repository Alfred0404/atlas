import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} — train loss: {train_loss:.4f}"
            )

            if self.val_loader is not None:
                val_loss = self._validate()
                logger.info(f"Epoch {epoch} — val loss: {val_loss:.4f}")
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=is_best)
            else:
                self._save_checkpoint(epoch, train_loss)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)  # (B, S, V)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"  step {self.global_step} — loss: {loss.item():.4f}, lr: {lr:.2e}"
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

            logits = self.model(x)
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

    @staticmethod
    def load_checkpoint(
        path: str, model: nn.Module, device: str = "cuda"
    ) -> dict:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
