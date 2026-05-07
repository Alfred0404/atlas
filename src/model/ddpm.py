import math
import torch


class DDPM:
    """DDPM noise schedule and forward/reverse diffusion utilities."""

    def __init__(self, T: int, schedule: str = "cosine", device: str = "cpu"):
        self.T = T
        betas = self._make_betas(T, schedule)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        self.betas = betas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = alpha_bar.sqrt().to(device)
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().to(device)
        self.sqrt_alphas = alphas.sqrt().to(device)
        # Posterior variance and mean coefficients for q(x_{t-1} | x_t, x_0)
        self.post_var = (betas * (1 - alpha_bar_prev) / (1 - alpha_bar)).to(device)
        self.post_coef1 = (alpha_bar_prev.sqrt() * betas / (1 - alpha_bar)).to(device)
        self.post_coef2 = (alphas.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)).to(device)

    @staticmethod
    def _make_betas(T: int, schedule: str) -> torch.Tensor:
        if schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, T, T + 1) / T
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return betas.clamp(0.0, 0.999)
        return torch.linspace(1e-4, 0.02, T)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x0 at timestep t.

        Args:
            x0: (B, N, 3) clean positions
            t:  (B,) timestep indices in [0, T-1]
        Returns:
            (xt, noise): noisy positions and the added noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        s1a = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sa * x0 + s1a * noise, noise

    @torch.no_grad()
    def p_sample(
        self,
        model,
        xt: torch.Tensor,
        t: int,
        padding_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """One reverse step: x_t → x_{t-1}.

        Returns:
            (x_{t-1}, model_output_dict)
        """
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=xt.device, dtype=torch.long)

        out = model(xt, t_batch, padding_mask)
        pred_noise = out["noise_pred"]

        # Reconstruct x_0 estimate, then compute posterior mean
        x0_pred = (
            xt - self.sqrt_one_minus_alpha_bar[t] * pred_noise
        ) / self.sqrt_alpha_bar[t]
        x0_pred = x0_pred.clamp(-3.0, 3.0)

        mean = self.post_coef1[t] * x0_pred + self.post_coef2[t] * xt

        if t == 0:
            return mean, out

        noise = torch.randn_like(xt)
        return mean + self.post_var[t].sqrt() * noise, out

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple,
        device: str,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """Full reverse diffusion: pure noise → generated positions.

        Args:
            model: DiT (eval mode)
            shape: (B, N, 3)
            device: torch device string
            padding_mask: (B, N) bool, True = padded
        Returns:
            dict with 'positions', 'part_logits', 'color_logits', 'rot_logits'
        """
        xt = torch.randn(shape, device=device)
        last_out = None
        for t in reversed(range(self.T)):
            xt, last_out = self.p_sample(model, xt, t, padding_mask)
        return {"positions": xt, **{k: v for k, v in last_out.items() if k != "noise_pred"}}
