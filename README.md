# Aerodynamic PINN

A PyTorch physics-informed neural network for steady, three-dimensional aerodynamic flow. The model learns velocity and pressure from OpenFOAM CSV exports at different reference velocities, then incorporates incompressible Navier–Stokes residuals during training.

The repository root is `ML`. Run the commands below from that directory.

## Setup
Training automatically uses CUDA when available and otherwise uses the CPU. Apple Metal/MPS is not selected by the current entry point. Current model uses user-edited OpenFoam files to properly train (deleting a few columns for concision).

`paraviewWallDist.txt` is a separate ParaView Programmable Filter script for checking wall distances and boundary conditions. It is not utilized yet as it is part of a future expansion to fully close the turbulence model in this model

## Data

The default cases use reference velocities of **33.27 m/s** and **82.6 m/s**. Each case also defines its longitudinal length, cross-stream length, and reference density in `CASE_FILES`.

The required CSV columns are mapped as follows:

| CSV column | Field |
| --- | --- |
| `Points_0` | `x` |
| `Points_1` | `y` |
| `Points_2` | `z` |
| `U_0` | `u` |
| `U_1` | `v` |
| `U_2` | `w` |
| `p` | Pressure |

The loader adds the reference velocity from `CASE_FILES`; it does not require a `u_ref` column in the CSV. Additional export columns can remain in the files. Update `COLUMN_MAP` if the export uses different names.

Data is physically nondimensionalized before training:

| Quantity | Scaling |
| --- | --- |
| Longitudinal coordinate | `x / l_ref` |
| Cross-stream coordinates | `y / cross_ref`, `z / cross_ref` |
| Case-conditioning input | `Uref / max(Uref across cases)` |
| Velocity targets | `u / Uref`, `v / Uref`, `w / Uref` |
| Pressure target | `p / Uref²` |

The residual treats the reconstructed pressure as kinematic pressure. New exports must use a compatible pressure convention.

The CSVs are loaded into memory together. The default split requests 50,000 training points and 50,000 validation points from the combined population, so provide enough eligible rows or reduce those settings.

`Data/` is excluded from Git. A fresh clone needs the CSV files supplied separately; mesh files and native OpenFOAM case dictionaries are not read by `main.py`.

## Run training

```bash
python -u main.py
```

Settings are edited directly in `config.py`; there are no command-line configuration arguments. Run from the repository root so dashboard files and checkpoints are written to the expected directories.

Training starts a dashboard server on port **8000** and attempts to open [http://localhost:8000/index.html](http://localhost:8000/index.html). Only one training process should use that port at a time. The server binds to all network interfaces.

## Model and training

The network maps four nondimensional inputs, `[x, y, z, u_ref]`, to four nondimensional outputs, `[u, v, w, p]`. Inputs are mapped internally to `[-1, 1]`. Spatial Fourier features augment the three coordinate inputs before an eight-layer, width-128 tanh network. The case-conditioning input remains separate from the Fourier features.

Training has two stages:

1. **Supervised pretraining:** epochs `0` through `PRETRAIN_EPOCHS`, inclusive. Each field has unit weight and the PDE coefficient is zero. PDE residuals are still evaluated periodically for diagnostics. The supervised training subset refreshes periodically while validation remains fixed.
2. **PDE training:** epochs greater than `PRETRAIN_EPOCHS`. The objective combines the supervised loss with momentum and continuity residuals. The PDE coefficient is controlled using supervised/PDE gradient norms and their smoothed cosine similarity, with a gradual cap on the intended weighted gradient ratio. This behaves closer to a commercial-grade CFD software, making small steps to reduce the physical residuals as PDE training progresses.

Supervised errors are divided by the initial training-target standard deviation of each field, clamped to at least `1e-6`, before MSE is calculated. These four scales are printed at startup and retained across training-subset refreshes. Stored targets and model outputs retain their physical nondimensional scaling. Dashboard field losses report unscaled MSE; the optimizer uses the scaled errors.

Supervised sampling mixes distributions over streamwise velocity deficit, signed-log crossflow scores, and pressure. Occupied ranges receive equal proposal mass within each field, and the field distributions are combined using `SUPERVISED_SAMPLING_FIELD_WEIGHTS`. Sampling scores do not transform the physical targets. Validation points are excluded from training selection.

PDE collocation uses a residual-weighted candidate pool within the model-input bounding box. Collocation points are distinct from supervised and validation points.

At learning-rate update epochs, fresh validation determines:

```text
learning_rate = LR × mean(relative_L2_u, relative_L2_v, relative_L2_w, relative_L2_p)
```

The relative errors are fractions. For example, a mean error of `0.35` gives a learning rate of `3.5e-5` when `LR = 1e-4`. Each update uses the configured initial `LR` as its base.

## Default configuration

| Setting | Default |
| --- | --- |
| `EPOCHS` | 500,000 |
| `PRETRAIN_EPOCHS` | 20,000 |
| `LR` | `1e-4` |
| `LR_UPDATE_EVERY` | 50 epochs |
| `N_TRAIN_POINTS` | 50,000 |
| `N_VAL_POINTS` | 50,000 |
| `N_COLLOCATION` | 8,000 |
| `PDE_RAMP_EPOCHS` | 15,000 |
| `PDE_MAX_GRAD_RATIO` | 0.3 |
| `PDE_GRADIENT_EVERY` | 50 epochs |
| `SUPERVISED_REFRESH_EVERY` | 250 epochs, during pretraining |
| `COLLOCATION_REFRESH_EVERY` | 200 epochs |
| `VALIDATION_EVERY` | 20 epochs, plus learning-rate update epochs |
| `PLOT_EVERY` | 100 epochs |
| `CHECKPOINT_SAVE_EVERY` | 100 epochs |
| `AUTO_RESUME` | `True` |

`PDE_MAX_GRAD_RATIO` controls the intended weighted PDE/data gradient ratio at controller updates; it is not a fraction of the scalar loss. `PDE_LR` and `PDE_PHASE_LR` are legacy settings and do not control the current learning-rate update rule.

## Dashboard and checkpoints

The dashboard reads generated JSON and a combined training plot. It displays losses, fieldwise validation relative L2 errors, prediction statistics, parity plots, wake/freestream errors, PDE residual distributions, gradient diagnostics, and supervised sampling distributions.

| Output | Purpose |
| --- | --- |
| `dashboard/status.json` | Latest metrics and training progress |
| `dashboard/alerts.json` | Runtime alert entries |
| `dashboard/training_monitor.png` | Combined diagnostic plots |
| `checkpoints/checkpoint_latest.pt` | Most recent saved training state |
| `checkpoints/checkpoint_previous.pt` | Previous saved state |
| `checkpoints/checkpoint_pde_start.pt` | Snapshot near the pretraining/PDE boundary |

With `AUTO_RESUME = True`, startup loads the latest checkpoint, falling back to the previous file if the latest is missing or cannot be deserialized. Checkpoints store model weights, optimizer state, loss-controller state, diagnostic histories, and the saved epoch. Training continues from the next epoch.

To start an independent run, preserve existing checkpoints in another directory and set `AUTO_RESUME = False`. Disabling resume alone does not change the output directory. Keep architecture and spatial-frequency settings compatible when resuming.

Non-finite predictions, newly calculated PDE residuals, total loss, or parameter gradients raise an error before the optimizer step. Interrupting training does not trigger an extra checkpoint save; restarting uses the most recent scheduled save.

Datasets, checkpoints, generated diagnostics, dashboard JSON/images, logs, and Python caches are excluded by [.gitignore](.gitignore).

## Source layout

| File | Responsibility |
| --- | --- |
| [main.py](main.py) | Assemble the dataset, model, optimizer, diagnostics, and trainer |
| [config.py](config.py) | Case metadata, column mapping, and training settings |
| [data.py](data.py) | CSV loading, nondimensionalization, sampling, and validation split |
| [model.py](model.py) | Tanh network and spatial Fourier features |
| [losses.py](losses.py) | Supervised loss, PDE residual, and PDE gradient controller |
| [colloc.py](colloc.py) | Residual-adaptive collocation sampling |
| [training.py](training.py) | Training loop, validation schedule, and learning-rate updates |
| [validation.py](validation.py) | Fieldwise and regional validation errors |
| [diagnostics.py](diagnostics.py) | Training and validation history buffers |
| [checkpoint.py](checkpoint.py) | Save, rotate, restore, and migrate checkpoints |
| [dashboard/](dashboard/) | Browser interface, JSON reporting, plots, and HTTP server |
| [paraviewWallDist.txt](paraviewWallDist.txt) | Separate ParaView wall-distance utility |

## Current scope

The PDE residual implements steady incompressible Navier–Stokes with molecular viscosity. The model predicts only `u`, `v`, `w`, and `p`; exported turbulence fields such as `nut`, `k`, and `omega` are not used to close the momentum equations. `USE_FROZEN_NUT_LAPLACIAN` is currently unused. The residual therefore does not enforce a complete RANS or k–omega SST closure.

Collocation samples the axis-aligned input bounding box without excluding solid geometry. There is no explicit wall or inlet boundary-condition loss, and time is not a model input. The PDE uses the first case's length scales and density, so cases should share those properties. Fixed validation points are drawn from the loaded cases; this split does not establish accuracy on a held-out geometry or reference velocity.
