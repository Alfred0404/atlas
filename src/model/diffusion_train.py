import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .dit import DiT
from .ddpm import DDPM
from .diffusion_config import DiffusionConfig
from ..utils.logging import setup_logging

logger = setup_logging()


class DiffusionTrainer:
    def __init__(self, cfg: DiffusionConfig, device: str):
        self.cfg = cfg
        self.device = device

        self.model = DiT(cfg).to(device)
        self.ddpm = DDPM(cfg.T, cfg.beta_schedule, device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max=max(1, cfg.max_epochs - cfg.warmup_epochs),
                                   eta_min=1e-5)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine],
                                      milestones=[cfg.warmup_epochs])

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.global_step = 0

    def _loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x0 = batch["positions"].to(self.device)           # (B, N, 3)
        padding_mask = batch["padding_mask"].to(self.device)  # (B, N)
        part_ids = batch["part_ids"].to(self.device)       # (B, N)
        color_ids = batch["color_ids"].to(self.device)     # (B, N)
        rot_ids = batch["rot_ids"].to(self.device)         # (B, N)

        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=self.device)

        xt, noise = self.ddpm.q_sample(x0, t)
        xt = xt.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        out = self.model(xt, t, padding_mask)

        # Continuous loss on non-padded bricks only
        valid = ~padding_mask  # (B, N)
        pos_loss = (
            ((noise - out["noise_pred"]) ** 2)
            .mean(dim=-1)       # (B, N)
            .masked_fill(~valid, 0.0)
            .sum() / valid.sum().clamp(min=1)
        )

        # Discrete losses — only at low t (positions are near-clean, classification is tractable)
        discrete_mask = t < self.cfg.discrete_t_max  # (B,) — True = include discrete loss

        def ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            targets = targets.masked_fill(padding_mask, -100)
            targets = targets.masked_fill(targets == 0, -100)          # exclude UNK (id=0)
            targets = targets.masked_fill(~discrete_mask.unsqueeze(1), -100)
            if not (targets != -100).any():
                return torch.zeros((), device=logits.device)
            return self.ce(logits.view(-1, logits.shape[-1]), targets.view(-1))

        part_loss = ce(out["part_logits"], part_ids)
        color_loss = ce(out["color_logits"], color_ids)
        rot_loss = ce(out["rot_logits"], rot_ids)

        cfg = self.cfg
        loss = (
            cfg.lambda_pos * pos_loss
            + cfg.lambda_part * part_loss
            + cfg.lambda_color * color_loss
            + cfg.lambda_rot * rot_loss
        )
        return loss, {
            "pos": pos_loss.item(),
            "part": part_loss.item(),
            "color": color_loss.item(),
            "rot": rot_loss.item(),
        }

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, metrics = self._loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        return {"loss": loss.item(), **metrics}

    def eval_step(self, batch: dict) -> dict:
        self.model.eval()
        with torch.no_grad():
            loss, metrics = self._loss(batch)
        return {"loss": loss.item(), **metrics}

    def save_checkpoint(self, ckpt_dir: Path):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "cfg": self.cfg,
            },
            ckpt_dir / "latest.pt",
        )
        logger.info("Checkpoint saved at step %d", self.global_step)

    def load_checkpoint(self, ckpt_dir: Path) -> bool:
        path = ckpt_dir / "latest.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt["global_step"]
        logger.info("Resumed from step %d", self.global_step)
        return True
