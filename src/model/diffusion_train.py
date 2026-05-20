import torch
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

        self.global_step = 0

    def _loss(self, batch: dict) -> torch.Tensor:
        x0 = batch["positions"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)
        part_ids = batch["part_ids"].to(self.device)
        color_ids = batch["color_ids"].to(self.device)
        rot_ids = batch["rot_ids"].to(self.device)

        B = x0.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=self.device)

        xt, noise = self.ddpm.q_sample(x0, t)
        xt = xt.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        pred = self.model(xt, t, part_ids, color_ids, rot_ids, padding_mask)["noise_pred"]

        valid = ~padding_mask
        loss = (
            ((noise - pred) ** 2)
            .mean(dim=-1)
            .masked_fill(~valid, 0.0)
            .sum() / valid.sum().clamp(min=1)
        )
        return loss

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss = self._loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        return {"loss": loss.item()}

    def eval_step(self, batch: dict) -> dict:
        self.model.eval()
        with torch.no_grad():
            loss = self._loss(batch)
        return {"loss": loss.item()}

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
