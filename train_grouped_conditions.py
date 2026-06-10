#!/usr/bin/env python3
"""Train shared MASBots models on all trials from each 2f/2h condition."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from train_all_masbots import discover_pairs
from train_masbots_cvae import (
    Config,
    DirectedInteractionCVAE,
    TrajectoryWindows,
    build_state,
    decode_raw_prediction,
    plot_losses,
    run_epoch,
)


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def normalize_group(states: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    flat = np.concatenate([state.reshape(-1, state.shape[-1]) for state in states], axis=0)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return [((state - mean) / std).astype(np.float32) for state in states], mean, std


def split_trial(dataset: TrajectoryWindows, val_fraction: float) -> tuple[Subset, Subset]:
    val_len = max(1, int(round(len(dataset) * val_fraction)))
    train_len = len(dataset) - val_len
    return Subset(dataset, range(train_len)), Subset(dataset, range(train_len, len(dataset)))


def denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def plot_trial_prediction(
    model: DirectedInteractionCVAE,
    dataset: Subset,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: Config,
    device: torch.device,
    title: str,
    out: Path,
) -> float:
    batch = dataset[0]
    past = batch["past"].unsqueeze(0).to(device)
    pair = batch["pair"].unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model.predict_mean(past, pair)
        pred_norm = decode_raw_prediction(raw, past, cfg).cpu().numpy().reshape(2, cfg.future, -1)

    past_seq = denorm(batch["past_seq"].numpy(), mean, std)
    true = denorm(batch["future_seq"].numpy(), mean, std)
    pred = denorm(pred_norm, mean, std)
    mse = float(np.mean((pred_norm - batch["future_seq"].numpy()) ** 2))

    colors = {
        "past": ["tab:cyan", "tab:olive"],
        "real": ["tab:blue", "tab:green"],
        "model": ["tab:red", "tab:purple"],
    }
    all_xy = np.concatenate([past_seq[:, :, :2], true[:, :, :2], pred[:, :, :2]], axis=1).reshape(-1, 2)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    pad = max(xmax - xmin, ymax - ymin, 1.0) * 0.08

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    specs = [
        ("Past context", [("past", past_seq, "-")]),
        ("Real future", [("real", true, "-")]),
        ("Shared-model prediction", [("model", pred, "--")]),
        ("Real and model together", [("real", true, "-"), ("model", pred, "--")]),
    ]
    for ax, (panel_title, series) in zip(axes.flat, specs):
        for label, arr, style in series:
            for bot in range(2):
                ax.plot(
                    arr[bot, :, 0],
                    arr[bot, :, 1],
                    color=colors[label][bot],
                    linestyle=style,
                    linewidth=1.8,
                    label=f"bot {bot + 1} {label}",
                )
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymax + pad, ymin - pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_title(panel_title)
    fig.suptitle(f"{title} | grouped condition model | val MSE={mse:.4f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return mse


def train_condition(
    condition: str,
    pairs: list[tuple[str, Path, Path]],
    args: argparse.Namespace,
    root: Path,
) -> dict:
    outdir = root / condition
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        big=str(pairs[0][1]),
        dot=str(pairs[0][2]),
        outdir=str(outdir),
        past=args.past,
        future=args.future,
        stride=args.stride,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        velocity_weight=args.velocity_weight,
        acceleration_weight=args.acceleration_weight,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        theta_mode=args.theta_mode,
        split_mode="chronological",
        plot_frames=args.future,
        output_mode=args.output_mode,
        prediction_smooth_window=args.prediction_smooth_window,
    )

    print(f"\nBuilding {condition} states from {len(pairs)} trials...", flush=True)
    raw_states = [build_state(str(big), str(dot), cfg.fps, cfg.theta_mode) for _, big, dot in pairs]
    states, mean, std = normalize_group(raw_states)
    datasets = [TrajectoryWindows(state, cfg.past, cfg.future, cfg.stride) for state in states]
    splits = [split_trial(dataset, cfg.val_fraction) for dataset in datasets]
    train_sets = [split[0] for split in splits]
    val_sets = [split[1] for split in splits]
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=cfg.batch_size, shuffle=False)

    sample = datasets[0][0]
    model = DirectedInteractionCVAE(
        past_dim=sample["past"].numel(),
        future_dim=sample["future"].numel(),
        pair_dim=sample["pair"].numel(),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        layers=cfg.layers,
    )
    device = torch.device("mps" if cfg.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if cfg.device != "auto":
        device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-5)

    best_val = math.inf
    stale = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(model, train_loader, optimizer, cfg.beta, cfg, device)
        val = run_epoch(model, val_loader, None, cfg.beta, cfg, device)
        scheduler.step(val["loss"])
        row = {f"train_{k}": v for k, v in train.items()} | {f"val_{k}": v for k, v in val.items()}
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        if val["loss"] < best_val:
            best_val = val["loss"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "condition": condition,
                    "trials": [{"name": n, "big": str(b), "dot": str(d)} for n, b, d in pairs],
                    "state_mean": mean,
                    "state_std": std,
                    "history": history,
                },
                outdir / "best_model.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{condition} epoch {epoch:04d} train={train['loss']:.5f} "
                f"val={val['loss']:.5f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if stale >= args.early_stopping:
            print(f"{condition}: early stopping at epoch {epoch}", flush=True)
            break

    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_losses(history, outdir)
    checkpoint = torch.load(outdir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    trial_results = []
    for (name, _, _), val_set in zip(pairs, val_sets):
        mse = plot_trial_prediction(
            model,
            val_set,
            mean,
            std,
            cfg,
            device,
            name,
            outdir / f"{slugify(name)}_grouped_prediction_{cfg.future}frames.png",
        )
        trial_results.append({"name": name, "val_prediction_mse": mse})

    result = {
        "condition": condition,
        "best_combined_val_loss": best_val,
        "epochs_trained": len(history),
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset),
        "trials": trial_results,
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="outputs/masbots_cvae/data")
    parser.add_argument("--out-root", default="outputs/masbots_cvae/runs/grouped_3000_delta_smooth61")
    parser.add_argument("--past", type=int, default=120)
    parser.add_argument("--future", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.5)
    parser.add_argument("--prediction-smooth-window", type=int, default=61)
    parser.add_argument("--output-mode", default="delta")
    parser.add_argument("--theta-mode", default="center")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pairs = discover_pairs(Path(args.data_root))
    groups = {
        "2f": [pair for pair in pairs if "_2f " in pair[0]],
        "2h": [pair for pair in pairs if "_2h " in pair[0]],
    }
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    results = [train_condition(condition, group, args, root) for condition, group in groups.items()]
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"{result['condition']}: best combined val loss={result['best_combined_val_loss']:.6f}, "
            f"epochs={result['epochs_trained']}"
        )


if __name__ == "__main__":
    main()
