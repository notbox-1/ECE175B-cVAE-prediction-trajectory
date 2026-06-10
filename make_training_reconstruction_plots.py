#!/usr/bin/env python3
"""Create real-vs-training reconstruction plots from saved cVAE runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "masbots_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import torch
from torch.utils.data import Subset, random_split

from train_masbots_cvae import (
    Config,
    DirectedInteractionCVAE,
    TrajectoryWindows,
    build_state,
    decode_raw_prediction,
    normalize_state,
)


def denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def split_dataset(dataset: TrajectoryWindows, cfg: Config):
    val_len = max(1, int(round(len(dataset) * cfg.val_fraction)))
    train_len = len(dataset) - val_len
    if cfg.split_mode == "chronological":
        return Subset(dataset, range(0, train_len)), Subset(dataset, range(train_len, len(dataset)))
    return random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))


def reconstruct_error(model, batch, cfg, device) -> float:
    past = batch["past"].unsqueeze(0).to(device)
    future = batch["future"].unsqueeze(0).to(device)
    pair = batch["pair"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model.reconstruct_mean(past, future, pair)
        pred = decode_raw_prediction(raw, past, cfg).cpu().numpy()
    return float(np.mean((pred.reshape(batch["future_seq"].shape) - batch["future_seq"].numpy()) ** 2))


def choose_training_sample(model, train_set, cfg, device, candidates: int):
    n = min(candidates, len(train_set))
    best_i = 0
    best_err = float("inf")
    for i in np.linspace(0, len(train_set) - 1, n, dtype=int):
        err = reconstruct_error(model, train_set[int(i)], cfg, device)
        if err < best_err:
            best_err = err
            best_i = int(i)
    return train_set[best_i], best_i, best_err


def set_common_axes(ax, arrays: list[np.ndarray]) -> None:
    xy = np.concatenate([a[:, :, :2].reshape(-1, 2) for a in arrays], axis=0)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    pad = max(xmax - xmin, ymax - ymin, 1.0) * 0.08
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymax + pad, ymin - pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def draw_paths(
    ax,
    arrays: list[np.ndarray],
    labels: list[str],
    colors: list[list[str]],
    styles: list[str],
    title: str,
    axis_arrays: list[np.ndarray],
) -> None:
    set_common_axes(ax, axis_arrays)
    for arr, label, bot_colors, style in zip(arrays, labels, colors, styles):
        for bot in range(arr.shape[0]):
            ax.plot(
                arr[bot, :, 0],
                arr[bot, :, 1],
                color=bot_colors[bot],
                linestyle=style,
                linewidth=2.0,
                label=f"bot {bot + 1} {label}",
            )
            ax.scatter(arr[bot, 0, 0], arr[bot, 0, 1], color=bot_colors[bot], s=18)
    ax.set_title(title)
    ax.legend(fontsize=7)


def save_four_panel(run_dir: Path, past_seq: np.ndarray, true: np.ndarray, pred: np.ndarray, frames: int) -> Path:
    past_colors = ["tab:cyan", "tab:olive"]
    real_colors = ["tab:blue", "tab:green"]
    model_colors = ["tab:red", "tab:purple"]
    axis_arrays = [past_seq, true, pred]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    draw_paths(axes[0, 0], [past_seq], ["past"], [past_colors], ["-"], "Past context", axis_arrays)
    draw_paths(axes[0, 1], [true], ["real"], [real_colors], ["-"], "Real future", axis_arrays)
    draw_paths(axes[1, 0], [pred], ["model"], [model_colors], ["--"], "Model training result", axis_arrays)
    draw_paths(
        axes[1, 1],
        [true, pred],
        ["real", "model"],
        [real_colors, model_colors],
        ["-", "--"],
        "Real and model together",
        axis_arrays,
    )
    fig.suptitle(f"Training reconstruction comparison ({frames} frames)", y=0.995)
    fig.tight_layout()
    out = run_dir / f"four_panel_training_{frames}frames.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_stage_animation(
    run_dir: Path,
    past_seq: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    frames: int,
    step: int,
    fps: int,
) -> Path:
    from matplotlib import animation

    past_colors = ["tab:cyan", "tab:olive"]
    real_colors = ["tab:blue", "tab:green"]
    model_colors = ["tab:red", "tab:purple"]
    axis_arrays = [past_seq, true, pred]

    stage_frames = []
    for stage, arr in [("past", past_seq), ("real", true), ("model", pred)]:
        for end in range(1, arr.shape[1] + 1, step):
            stage_frames.append((stage, end))
        if stage_frames[-1] != (stage, arr.shape[1]):
            stage_frames.append((stage, arr.shape[1]))

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame_info):
        stage, end = frame_info
        ax.clear()
        set_common_axes(ax, axis_arrays)
        if stage == "past":
            for bot in range(past_seq.shape[0]):
                ax.plot(past_seq[bot, :end, 0], past_seq[bot, :end, 1], color=past_colors[bot], linewidth=2.4)
            ax.set_title("Past trajectory")
        elif stage == "real":
            draw_paths(ax, [past_seq], ["past"], [past_colors], ["-"], "", axis_arrays)
            for bot in range(true.shape[0]):
                ax.plot(true[bot, :end, 0], true[bot, :end, 1], color=real_colors[bot], linewidth=2.4)
            ax.set_title("Real future trajectory")
        else:
            draw_paths(ax, [past_seq], ["past"], [past_colors], ["-"], "", axis_arrays)
            for bot in range(pred.shape[0]):
                ax.plot(pred[bot, :end, 0], pred[bot, :end, 1], color=model_colors[bot], linestyle="--", linewidth=2.4)
            ax.set_title("Model training trajectory")
        ax.text(0.02, 0.02, f"{stage}: {min(end, frames)} frames", transform=ax.transAxes)
        return []

    ani = animation.FuncAnimation(fig, update, frames=stage_frames, interval=1000 / fps, blit=False)
    out = run_dir / f"trajectory_animation_training_{frames}frames.gif"
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def plot_training_reconstruction(run_dir: Path, candidates: int, device_name: str, animation_step: int, fps: int) -> dict:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    cfg_data = checkpoint["config"]
    cfg_data.setdefault("plot_frames", cfg_data.get("future", 300))
    cfg_data.setdefault("split_mode", "chronological")
    cfg_data.setdefault("velocity_weight", 0.25)
    cfg_data.setdefault("acceleration_weight", 0.05)
    cfg_data.setdefault("output_mode", "absolute")
    cfg_data.setdefault("prediction_smooth_window", 1)
    cfg = Config(**cfg_data)
    cfg = replace(cfg, plot_frames=min(cfg.plot_frames, cfg.future))

    device = torch.device(device_name)
    state, state_mean, state_std = normalize_state(build_state(cfg.big, cfg.dot, cfg.fps, cfg.theta_mode))
    dataset = TrajectoryWindows(state, cfg.past, cfg.future, cfg.stride)
    train_set, _ = split_dataset(dataset, cfg)
    sample = dataset[0]
    model = DirectedInteractionCVAE(
        past_dim=sample["past"].numel(),
        future_dim=sample["future"].numel(),
        pair_dim=sample["pair"].numel(),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        layers=cfg.layers,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    batch, sample_index, err = choose_training_sample(model, train_set, cfg, device, candidates)
    past = batch["past"].unsqueeze(0).to(device)
    future = batch["future"].unsqueeze(0).to(device)
    pair = batch["pair"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model.reconstruct_mean(past, future, pair)
        pred = decode_raw_prediction(raw, past, cfg).cpu().numpy().reshape(2, cfg.future, -1)

    frames = min(cfg.plot_frames, cfg.future)
    true = denorm(batch["future_seq"].numpy(), state_mean, state_std)[:, :frames]
    pred = denorm(pred, state_mean, state_std)[:, :frames]
    past_seq = denorm(batch["past_seq"].numpy(), state_mean, state_std)

    panel = save_four_panel(run_dir, past_seq, true, pred, frames)
    gif = save_stage_animation(run_dir, past_seq, true, pred, frames, animation_step, fps)
    return {
        "run": run_dir.name,
        "plot": str(panel),
        "animation": str(gif),
        "sample_index": sample_index,
        "train_recon_mse": err,
    }


def make_contact_sheet(rows: list[dict], out_root: Path) -> Path:
    from PIL import Image, ImageDraw

    thumbs = []
    for row in rows:
        img = Image.open(row["plot"]).convert("RGB")
        img.thumbnail((520, 520))
        canvas = Image.new("RGB", (560, 610), "white")
        canvas.paste(img, ((560 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 535), row["run"][:48], fill="black")
        draw.text((16, 560), f"training recon MSE {row['train_recon_mse']:.5f}", fill="black")
        thumbs.append(canvas)

    cols = 2
    sheet_rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 560, sheet_rows * 610), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 560, (i // cols) * 610))
    out = out_root / "all_four_panel_training_300frames_contact_sheet.png"
    sheet.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="outputs/masbots_cvae/runs/all_local_300")
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--animation-step", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.runs_root)
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "best_model.pt").exists())
    rows = [plot_training_reconstruction(p, args.candidates, args.device, args.animation_step, args.fps) for p in run_dirs]
    (root / "training_reconstruction_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    sheet = make_contact_sheet(rows, root)
    print(f"Wrote {len(rows)} training reconstruction plots")
    print(sheet)


if __name__ == "__main__":
    main()
