#!/usr/bin/env python3
"""Core MASBots cVAE model and single-trial training pipeline.

This file loads paired big-circle and small-dot tracking data, constructs robot
states and directed interaction features, trains the cVAE, and saves the model,
training history, loss curves, and trajectory predictions for one trial.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "masbots_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset, random_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for training. Install it with `pip install torch`, "
        "or run `pip install -r requirements.txt` from this folder."
    ) from exc


@dataclass
class Config:
    big: str
    dot: str
    outdir: str = "runs/masbots_cvae"
    fps: float = 30.0
    past: int = 60
    future: int = 30
    stride: int = 5
    latent_dim: int = 16
    hidden_dim: int = 256
    layers: int = 3
    beta: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    velocity_weight: float = 0.25
    acceleration_weight: float = 0.05
    val_fraction: float = 0.2
    seed: int = 7
    device: str = "auto"
    theta_mode: str = "center"
    split_mode: str = "chronological"
    plot_frames: int = 300
    output_mode: str = "absolute"
    prediction_smooth_window: int = 1


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(path)
    if "Xmat" in data and "Ymat" in data:
        x_key, y_key = "Xmat", "Ymat"
    elif "X1" in data and "Y1" in data:
        x_key, y_key = "X1", "Y1"
    else:
        keys = ", ".join(k for k in data if not k.startswith("__"))
        raise KeyError(f"{path} does not contain Xmat/Ymat or X1/Y1. Found: {keys}")
    x = np.asarray(data[x_key], dtype=np.float64)
    y = np.asarray(data[y_key], dtype=np.float64)
    x[x == 0] = np.nan
    y[y == 0] = np.nan
    return x, y


def fit_circle_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    def residuals(c: np.ndarray) -> np.ndarray:
        cx, cy, r = c
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r

    cx0 = np.nanmean(x)
    cy0 = np.nanmean(y)
    r0 = np.nanmean(np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2))
    return tuple(least_squares(residuals, x0=[cx0, cy0, r0]).x)


def compute_icr_circle_fit(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int = 11,
    poly_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n_particles, n_frames = x.shape
    cx = np.full_like(x, np.nan, dtype=float)
    cy = np.full_like(y, np.nan, dtype=float)
    half_win = window_size // 2

    for i in range(n_particles):
        for t in range(n_frames):
            start = max(t - half_win, 0)
            end = min(t + half_win + 1, n_frames)
            valid = ~np.isnan(x[i, start:end]) & ~np.isnan(y[i, start:end])
            if np.sum(valid) < 3:
                continue
            try:
                cxc, cyc, _ = fit_circle_least_squares(
                    x[i, start:end][valid],
                    y[i, start:end][valid],
                )
                cx[i, t] = cxc
                cy[i, t] = cyc
            except ValueError:
                continue

        valid = ~np.isnan(cx[i]) & ~np.isnan(cy[i])
        idx = np.where(valid)[0]
        win = poly_smooth * 2 + 1
        if len(idx) > win:
            cx[i, idx] = savgol_filter(cx[i, idx], window_length=win, polyorder=poly_smooth)
            cy[i, idx] = savgol_filter(cy[i, idx], window_length=win, polyorder=poly_smooth)

    return cx, cy


def mapping_circle(xd: np.ndarray, yd: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[int]:
    n = x.shape[0]
    if n != 2:
        raise ValueError("This sample implementation expects exactly two tracked bots.")

    dist = np.full((2, 2), np.nan)
    for i in range(2):
        for j in range(2):
            dist[i, j] = np.nanmean(np.sqrt((xd[j] - x[i]) ** 2 + (yd[j] - y[i]) ** 2))
    return [0, 1] if dist[0, 0] + dist[1, 1] <= dist[0, 1] + dist[1, 0] else [1, 0]


def odd_window(max_window: int, n: int) -> int:
    win = min(max_window, n if n % 2 == 1 else n - 1)
    return max(win, 5)


def compute_theta_omega(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    fps: float,
    theta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_particles, n_frames = x.shape
    dot_to_bot = mapping_circle(xd, yd, x, y)

    theta = np.full((n_particles, n_frames), np.nan)
    if theta_mode == "center":
        for i in range(n_particles):
            dot_idx = dot_to_bot[i]
            dx = xd[dot_idx] - x[i]
            dy = yd[dot_idx] - y[i]
            valid = ~np.isnan(dx) & ~np.isnan(dy)
            theta[i, valid] = np.arctan2(dy[valid], dx[valid])
    elif theta_mode == "icr":
        cx, cy = compute_icr_circle_fit(xd, yd)
        icr_to_bot = mapping_circle(cx, cy, x, y)
        for i in range(n_particles):
            dot_idx = dot_to_bot[i]
            icr_idx = icr_to_bot[i]
            dx = xd[dot_idx] - cx[icr_idx]
            dy = yd[dot_idx] - cy[icr_idx]
            valid = ~np.isnan(dx) & ~np.isnan(dy)
            theta[i, valid] = np.arctan2(dy[valid], dx[valid])
    else:
        raise ValueError("--theta-mode must be either center or icr")

    theta_unwrap = theta.copy()
    for i in range(n_particles):
        valid = ~np.isnan(theta[i])
        if np.sum(valid) > 0:
            theta_unwrap[i, valid] = np.unwrap(theta[i, valid])

    theta_smooth = theta_unwrap.copy()
    for i in range(n_particles):
        idx = np.where(~np.isnan(theta_unwrap[i]))[0]
        if len(idx) > 7:
            win = odd_window(201, len(idx))
            theta_smooth[i, idx] = savgol_filter(theta_unwrap[i, idx], window_length=win, polyorder=2)

    t = np.arange(n_frames) / fps
    omega = np.full((n_particles, n_frames), np.nan)
    for i in range(n_particles):
        idx = np.where(~np.isnan(theta_smooth[i]))[0]
        if len(idx) > 2:
            omega[i, idx] = np.gradient(theta_smooth[i, idx], t[idx])

    omega[omega <= 0] = np.nan
    omega_smooth = omega.copy()
    for i in range(n_particles):
        idx = np.where(~np.isnan(omega[i]))[0]
        if len(idx) > 7:
            win = odd_window(201, len(idx))
            omega_smooth[i, idx] = savgol_filter(omega[i, idx], window_length=win, polyorder=2)

    return theta_smooth, omega_smooth


def fill_nan_timewise(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for idx in np.ndindex(out.shape[:-1]):
        series = out[idx]
        valid = ~np.isnan(series)
        if np.any(valid):
            xs = np.arange(series.size)
            series[~valid] = np.interp(xs[~valid], xs[valid], series[valid])
        else:
            series[:] = 0.0
    return out


def build_state(big_path: str, dot_path: str, fps: float, theta_mode: str) -> np.ndarray:
    x, y = load_xy(big_path)
    xd, yd = load_xy(dot_path)
    theta, omega = compute_theta_omega(x, y, xd, yd, fps, theta_mode)

    state = np.stack(
        [
            x,
            y,
            np.cos(theta),
            np.sin(theta),
            omega,
        ],
        axis=-1,
    )
    return fill_nan_timewise(state).astype(np.float32)


def normalize_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = state.reshape(-1, state.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((state - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def directed_features(window: np.ndarray) -> np.ndarray:
    """Build f_ij != f_ji from a normalized state window with shape (N, T, F)."""
    pos = window[:, :, 0:2]
    heading = window[:, :, 2:4]
    omega = window[:, :, 4:5]
    vel = np.gradient(pos, axis=1)
    pieces = []
    n_agents = window.shape[0]
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            rel = np.concatenate(
                [
                    pos[j] - pos[i],
                    vel[j] - vel[i],
                    heading[j] - heading[i],
                    omega[j] - omega[i],
                ],
                axis=-1,
            )
            pieces.append(rel.reshape(-1))
    return np.concatenate(pieces).astype(np.float32)


class TrajectoryWindows(Dataset):
    def __init__(self, state: np.ndarray, past: int, future: int, stride: int):
        self.state = state
        self.past = past
        self.future = future
        total = state.shape[1]
        self.starts = list(range(0, total - past - future + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        past = self.state[:, start : start + self.past]
        future = self.state[:, start + self.past : start + self.past + self.future]
        pair = directed_features(past)
        return {
            "past": torch.from_numpy(past.reshape(-1)),
            "future": torch.from_numpy(future.reshape(-1)),
            "pair": torch.from_numpy(pair),
            "past_seq": torch.from_numpy(past),
            "future_seq": torch.from_numpy(future),
        }


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        blocks += [nn.Linear(dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


class DirectedInteractionCVAE(nn.Module):
    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        pair_dim: int,
        latent_dim: int,
        hidden_dim: int,
        layers: int,
    ):
        super().__init__()
        self.encoder = mlp(past_dim + future_dim + pair_dim, hidden_dim, hidden_dim, layers)
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.prior = mlp(past_dim + pair_dim, hidden_dim, hidden_dim, layers)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = mlp(past_dim + pair_dim + latent_dim, hidden_dim, future_dim, layers)

    def encode(self, past: torch.Tensor, future: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([past, future, pair], dim=-1))
        return self.posterior_mu(h), self.posterior_logvar(h).clamp(-8.0, 8.0)

    def prior_dist(self, past: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.prior(torch.cat([past, pair], dim=-1))
        return self.prior_mu(h), self.prior_logvar(h).clamp(-8.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, past: torch.Tensor, pair: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([past, pair, z], dim=-1))

    def forward(self, past: torch.Tensor, future: torch.Tensor, pair: torch.Tensor) -> dict[str, torch.Tensor]:
        q_mu, q_logvar = self.encode(past, future, pair)
        p_mu, p_logvar = self.prior_dist(past, pair)
        z = self.reparameterize(q_mu, q_logvar)
        pred = self.decode(past, pair, z)
        return {
            "pred": pred,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }

    def sample(self, past: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        p_mu, p_logvar = self.prior_dist(past, pair)
        z = self.reparameterize(p_mu, p_logvar)
        return self.decode(past, pair, z)

    def predict_mean(self, past: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        p_mu, _ = self.prior_dist(past, pair)
        return self.decode(past, pair, p_mu)

    def reconstruct_mean(self, past: torch.Tensor, future: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        q_mu, _ = self.encode(past, future, pair)
        return self.decode(past, pair, q_mu)


def kl_normal(q_mu: torch.Tensor, q_logvar: torch.Tensor, p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        p_logvar
        - q_logvar
        + (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) / torch.exp(p_logvar)
        - 1.0
    )


def decode_raw_prediction(raw: torch.Tensor, past: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.output_mode == "absolute":
        pred_seq = raw.view(raw.shape[0], 2, cfg.future, 5)
    elif cfg.output_mode == "delta":
        raw_seq = raw.view(raw.shape[0], 2, cfg.future, 5)
        past_seq = past.view(past.shape[0], 2, cfg.past, 5)
        pred_seq = raw_seq.clone()
        last_xy = past_seq[:, :, -1:, :2]
        pred_seq[:, :, :, :2] = last_xy + torch.cumsum(raw_seq[:, :, :, :2], dim=2)
    else:
        raise ValueError("--output-mode must be either absolute or delta")

    window = int(cfg.prediction_smooth_window)
    if window > 1:
        if window % 2 == 0:
            window += 1
        pad = window // 2
        xy = pred_seq[:, :, :, :2].permute(0, 1, 3, 2).reshape(-1, 1, cfg.future)
        xy = nn.functional.pad(xy, (pad, pad), mode="replicate")
        xy = nn.functional.avg_pool1d(xy, kernel_size=window, stride=1)
        xy = xy.reshape(pred_seq.shape[0], 2, 2, cfg.future).permute(0, 1, 3, 2)
        pred_seq = pred_seq.clone()
        pred_seq[:, :, :, :2] = xy
    return pred_seq.reshape(raw.shape[0], -1)


def trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    future_len: int,
    n_agents: int,
    n_features: int,
    velocity_weight: float,
    acceleration_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_seq = pred.view(pred.shape[0], n_agents, future_len, n_features)
    target_seq = target.view(target.shape[0], n_agents, future_len, n_features)
    recon = nn.functional.mse_loss(pred_seq, target_seq)

    pred_vel = pred_seq[:, :, 1:, :2] - pred_seq[:, :, :-1, :2]
    target_vel = target_seq[:, :, 1:, :2] - target_seq[:, :, :-1, :2]
    vel = nn.functional.mse_loss(pred_vel, target_vel)

    if future_len > 2:
        pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
        target_acc = target_vel[:, :, 1:] - target_vel[:, :, :-1]
        acc = nn.functional.mse_loss(pred_acc, target_acc)
    else:
        acc = torch.zeros((), device=pred.device)

    smooth = vel + acc
    total = recon + velocity_weight * vel + acceleration_weight * acc
    return total, recon, smooth


def run_epoch(
    model: DirectedInteractionCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "recon": 0.0, "smooth": 0.0, "kl": 0.0}
    n = 0
    for batch in loader:
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        pair = batch["pair"].to(device)
        with torch.set_grad_enabled(training):
            out = model(past, future, pair)
            pred = decode_raw_prediction(out["pred"], past, cfg)
            traj, recon, accel = trajectory_loss(
                pred,
                future,
                future_len=cfg.future,
                n_agents=2,
                n_features=5,
                velocity_weight=cfg.velocity_weight,
                acceleration_weight=cfg.acceleration_weight,
            )
            kl = kl_normal(out["q_mu"], out["q_logvar"], out["p_mu"], out["p_logvar"])
            loss = traj + beta * kl
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        bs = past.shape[0]
        for key, value in [("loss", loss), ("recon", recon), ("smooth", accel), ("kl", kl)]:
            totals[key] += float(value.detach().cpu()) * bs
        n += bs
    return {key: value / max(n, 1) for key, value in totals.items()}


def plot_losses(history: list[dict[str, float]], outdir: Path) -> None:
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, ["loss", "recon", "smooth"]):
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="val")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=180)
    plt.close(fig)


def plot_prediction(
    model: DirectedInteractionCVAE,
    dataset: Dataset,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cfg: Config,
    device: torch.device,
    outdir: Path,
) -> None:
    model.eval()
    batch = dataset[0]
    past = batch["past"].unsqueeze(0).to(device)
    pair = batch["pair"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model.predict_mean(past, pair)
        pred = decode_raw_prediction(raw, past, cfg).cpu().numpy().reshape(2, cfg.future, -1)
    past_seq = batch["past_seq"].numpy()
    true = batch["future_seq"].numpy()

    def denorm(x: np.ndarray) -> np.ndarray:
        return x * state_std + state_mean

    past_seq = denorm(past_seq)
    true = denorm(true)
    pred = denorm(pred)
    frames = min(cfg.plot_frames, cfg.future)
    true = true[:, :frames]
    pred = pred[:, :frames]

    fig, ax = plt.subplots(figsize=(7, 7))
    real_colors = ["tab:blue", "tab:green"]
    pred_colors = ["tab:red", "tab:purple"]
    for i in range(true.shape[0]):
        ax.plot(
            past_seq[i, :, 0],
            past_seq[i, :, 1],
            color="0.65",
            alpha=0.45,
            linewidth=1.5,
            label=f"bot {i + 1} past" if i == 0 else None,
        )
        ax.plot(true[i, :, 0], true[i, :, 1], color=real_colors[i], linewidth=2.2, label=f"bot {i + 1} real")
        ax.plot(
            pred[i, :, 0],
            pred[i, :, 1],
            color=pred_colors[i],
            linestyle="--",
            linewidth=2.2,
            label=f"bot {i + 1} model",
        )
        ax.scatter(true[i, 0, 0], true[i, 0, 1], color=real_colors[i], marker="o", s=25)
        ax.scatter(pred[i, 0, 0], pred[i, 0, 1], color=pred_colors[i], marker="x", s=35)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Real vs model trajectory ({frames} frames)")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_example.png", dpi=180)
    fig.savefig(outdir / f"real_vs_model_{frames}frames.png", dpi=220)
    plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in Config.__dataclass_fields__.items():
        default = value.default
        arg_type = type(default) if default is not None else str
        if field in {"big", "dot"}:
            parser.add_argument(f"--{field}", required=True)
        else:
            parser.add_argument(f"--{field.replace('_', '-')}", default=default, type=arg_type)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("mps" if cfg.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if cfg.device not in {"auto", "mps", "cpu", "cuda"}:
        raise ValueError("--device must be one of auto, mps, cpu, cuda")
    if cfg.device in {"mps", "cpu", "cuda"}:
        device = torch.device(cfg.device)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    state, state_mean, state_std = normalize_state(build_state(cfg.big, cfg.dot, cfg.fps, cfg.theta_mode))
    dataset = TrajectoryWindows(state, cfg.past, cfg.future, cfg.stride)
    if len(dataset) < 4:
        raise ValueError("Not enough trajectory windows. Reduce --past, --future, or --stride.")

    val_len = max(1, int(round(len(dataset) * cfg.val_fraction)))
    train_len = len(dataset) - val_len
    if cfg.split_mode == "chronological":
        train_set = Subset(dataset, range(0, train_len))
        val_set = Subset(dataset, range(train_len, len(dataset)))
    elif cfg.split_mode == "random":
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))
    else:
        raise ValueError("--split-mode must be either chronological or random")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    sample = dataset[0]
    model = DirectedInteractionCVAE(
        past_dim=sample["past"].numel(),
        future_dim=sample["future"].numel(),
        pair_dim=sample["pair"].numel(),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        layers=cfg.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = math.inf
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(model, train_loader, optimizer, cfg.beta, cfg, device)
        val = run_epoch(model, val_loader, None, cfg.beta, cfg, device)
        row = {f"train_{k}": v for k, v in train.items()} | {f"val_{k}": v for k, v in val.items()}
        history.append(row)
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "history": history,
                },
                outdir / "best_model.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:04d} "
                f"train loss={train['loss']:.6f} recon={train['recon']:.6f} kl={train['kl']:.6f} "
                f"val loss={val['loss']:.6f} recon={val['recon']:.6f} kl={val['kl']:.6f}"
            )

    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_losses(history, outdir)
    checkpoint = torch.load(outdir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    plot_prediction(model, val_set, state_mean, state_std, cfg, device, outdir)
    print(f"Saved run artifacts to {outdir}")


if __name__ == "__main__":
    main()
