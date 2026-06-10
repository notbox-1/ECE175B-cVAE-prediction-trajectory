# Final MASBots 3000-Frame Training Code

This folder contains one clean copy of every script needed for the final
independent-trial and grouped-condition 3000-frame training runs.

## Files

- `train_masbots_cvae.py`: shared cVAE model, preprocessing, losses, and single-pair trainer
- `train_all_masbots.py`: trains one separate model for every trial
- `train_grouped_conditions.py`: cross-trains one shared model on all three `2f` trials and one shared model on all three `2h` trials
- `make_training_reconstruction_plots.py`: creates four-panel trajectory plots and GIF animations from separate-trial checkpoints
- `requirements.txt`: Python dependencies

The expected data layout is:

```text
data/
  trial_name_2f.../
    ..._big circle.mat
    ..._small dots.mat
  trial_name_2h.../
    ..._big circle.mat
    ..._small dots.mat
```

## Environment

From the original workspace:

```bash
work/masbots_torch_env/bin/python -m pip install -r outputs/final_3000_training_code/requirements.txt
```

## Final Independent-Trial 3000-Frame Run

This trains six separate models, one for each trial:

```bash
work/masbots_torch_env/bin/python outputs/final_3000_training_code/train_all_masbots.py \
  --data-root outputs/masbots_cvae/data \
  --out-root outputs/masbots_cvae/runs/all_local_3000_delta_smooth61 \
  --epochs 120 \
  --past 120 \
  --future 3000 \
  --stride 50 \
  --batch-size 32 \
  --theta-mode center \
  --split-mode chronological \
  --device auto \
  --velocity-weight 1.0 \
  --acceleration-weight 0.5 \
  --output-mode delta \
  --prediction-smooth-window 61
```

Key settings:

- Predicts 3000 future frames from 120 past frames
- Predicts position changes rather than absolute positions
- Integrates predicted changes to reconstruct future positions
- Applies a 61-frame internal smoothing window
- Uses velocity and acceleration consistency losses

## Grouped/Cross-Training 3000-Frame Run

This trains one shared model using all three `2f` trials and another shared
model using all three `2h` trials:

```bash
work/masbots_torch_env/bin/python outputs/final_3000_training_code/train_grouped_conditions.py \
  --data-root outputs/masbots_cvae/data \
  --out-root outputs/masbots_cvae/runs/grouped_3000_delta_smooth61 \
  --epochs 220 \
  --past 120 \
  --future 3000 \
  --stride 50 \
  --batch-size 32 \
  --velocity-weight 1.0 \
  --acceleration-weight 0.5 \
  --output-mode delta \
  --prediction-smooth-window 61 \
  --early-stopping 35 \
  --device auto
```

The grouped trainer:

- Uses global normalization within each `2f` or `2h` condition
- Creates chronological training and validation windows from every trial
- Uses learning-rate reduction and early stopping
- Saves one `best_model.pt` for each condition
- Generates a grouped prediction plot for every trial

## Evaluation Note

The training loss combines normalized full-state MSE, velocity consistency,
acceleration consistency, and KL divergence. For final trajectory comparisons,
use raw-coordinate ADE, FDE, and position RMSE; do not interpret normalized
training loss as a classification-style accuracy rate.
