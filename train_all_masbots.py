#!/usr/bin/env python3
"""Launch independent cVAE training for every MASBots trial.

This file discovers each paired big-circle/small-dot dataset, calls
train_masbots_cvae.py once per trial, and writes a summary of the independently
trained models and their best validation losses.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "run"


def discover_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        mats = sorted(folder.glob("*.mat"))
        big = [p for p in mats if "big" in p.name.lower()]
        dot = [p for p in mats if "small" in p.name.lower() or "dot" in p.name.lower()]
        if len(big) != 1 or len(dot) != 1:
            print(f"Skipping {folder.name}: expected 1 big file and 1 dot file, found {len(big)} and {len(dot)}")
            continue
        pairs.append((folder.name, big[0], dot[0]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="outputs/masbots_cvae/data")
    parser.add_argument("--out-root", default="outputs/masbots_cvae/runs/all_local_300")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--past", type=int, default=60)
    parser.add_argument("--future", type=int, default=300)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.05)
    parser.add_argument("--theta-mode", default="center")
    parser.add_argument("--split-mode", default="chronological")
    parser.add_argument("--output-mode", default="absolute")
    parser.add_argument("--prediction-smooth-window", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(data_root)
    if not pairs:
        raise SystemExit(f"No trainable pairs found under {data_root}")

    summary = []
    train_script = Path(__file__).with_name("train_masbots_cvae.py")
    for index, (name, big, dot) in enumerate(pairs, start=1):
        outdir = out_root / slugify(name)
        print(f"\n=== [{index}/{len(pairs)}] Training {name} ===", flush=True)
        cmd = [
            sys.executable,
            str(train_script),
            "--big",
            str(big),
            "--dot",
            str(dot),
            "--outdir",
            str(outdir),
            "--epochs",
            str(args.epochs),
            "--past",
            str(args.past),
            "--future",
            str(args.future),
            "--stride",
            str(args.stride),
            "--batch-size",
            str(args.batch_size),
            "--velocity-weight",
            str(args.velocity_weight),
            "--acceleration-weight",
            str(args.acceleration_weight),
            "--theta-mode",
            args.theta_mode,
            "--split-mode",
            args.split_mode,
            "--output-mode",
            args.output_mode,
            "--prediction-smooth-window",
            str(args.prediction_smooth_window),
            "--device",
            args.device,
            "--hidden-dim",
            str(args.hidden_dim),
            "--latent-dim",
            str(args.latent_dim),
            "--plot-frames",
            str(args.future),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(cmd, check=True)
        history = json.loads((outdir / "history.json").read_text(encoding="utf-8"))
        best_epoch = min(range(len(history)), key=lambda i: history[i]["val_loss"]) + 1
        best = history[best_epoch - 1]
        summary.append(
            {
                "name": name,
                "big": str(big),
                "dot": str(dot),
                "outdir": str(outdir),
                "best_epoch": best_epoch,
                "best_val_loss": best["val_loss"],
                "best_val_recon": best["val_recon"],
                "plot": str(outdir / f"real_vs_model_{args.future}frames.png"),
            }
        )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished {len(summary)} runs. Summary saved to {summary_path}")
    for row in summary:
        print(f"{row['name']}: best val_loss={row['best_val_loss']:.6f} at epoch {row['best_epoch']}")


if __name__ == "__main__":
    main()
