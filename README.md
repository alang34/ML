# Aerodynamic PINN

A PyTorch physics-informed neural network for steady, three-dimensional aerodynamic flow. The model learns velocity and pressure from OpenFOAM CSV exports at different reference velocities, then incorporates incompressible Navier–Stokes residuals during training.

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

`Data/` is excluded from Git. Supply the CSV files and keep each case's ASCII OpenFOAM `constant/polyMesh/{boundary,points,faces}` and `0/U` next to its CSV. The boundary reader accepts plain and gzip-compressed files. For CSV-only runs, set both `BOUNDARY_WARMUP_STEPS` and `BOUNDARY_LOSS_WEIGHT` to zero.

## Run training

```bash
python -u main.py
```

Settings are edited directly in `config.py`; there are no command-line configuration arguments. Run from the repository root so dashboard files and checkpoints are written to the expected directories.

Training starts a dashboard server on port **8000** and attempts to open [http://localhost:8000/index.html](http://localhost:8000/index.html). Only one training process should use that port at a time. The server binds to all network interfaces.

## Model and training

The network maps four nondimensional inputs, `[x, y, z, u_ref]`, to four nondimensional outputs, `[u, v, w, p]`. Inputs are mapped internally to `[-1, 1]`. Spatial Fourier features augment the three coordinate inputs before an eight-layer, width-128 tanh network. The case-conditioning input remains separate from the Fourier features.

The raw network outputs are vector potentials `[ax, ay, az]` and normalized pressure. With `X = x/l_ref`, `Y = y/cross_ref`, `Z = z/cross_ref`, and `beta = l_ref/cross_ref`, `model.forward()` reconstructs the normalized velocities:

```text
u = 1 + beta * (d(az)/dY - d(ay)/dZ)
v = beta * d(ax)/dZ - d(az)/dX
w = d(ay)/dX - beta * d(ax)/dY
```

This enforces `du/dX + beta * (dv/dY + dw/dZ) = 0` up to floating-point precision. Reference lengths must match across cases. Supervised targets and validation use `[u, v, w, p]`. Forward prediction evaluates the curl using the chain rule for the Fourier features and tanh layers, so inference works without building an autograd graph. Momentum training still differentiates through the curl.

Fresh runs initialize the velocity field with a boundary-only warm-up. Slip patches penalize only normal velocity. Stationary no-slip walls penalize all three components, and fixed-value patches use the specified velocity divided by the reference speed. Faces are sampled by area within each case and condition group, and the groups receive equal weight. The mixed inlet/outlet patch is excluded from this prescribed-velocity loss.

After warm-up, the weighted boundary loss contributes separately in both training stages and appears on the dashboard. Resumed runs continue with the boundary loss from their saved weights and skip the initializer.

Training has two numbered epoch stages:

1. **Supervised pretraining:** epochs `0` through `PRETRAIN_EPOCHS`, inclusive. Each field has unit weight and the PDE coefficient is zero. PDE residuals are still evaluated periodically for diagnostics. The supervised training subset refreshes periodically while validation remains fixed.
2. **PDE training:** epochs greater than `PRETRAIN_EPOCHS`. The objective combines the supervised loss with a smooth-L1 loss averaged over the three momentum residuals. Continuity remains a diagnostic because the output construction already enforces it. The PDE coefficient is controlled using supervised/PDE gradient norms and their smoothed cosine similarity, with a gradual cap on the intended weighted gradient ratio.

Supervised errors are divided by the initial training-target standard deviation of each field, clamped to at least `1e-6`, before MSE is calculated. These four scales are printed at startup and retained across training-subset refreshes. Stored targets and model outputs retain their physical nondimensional scaling. Dashboard field losses report unscaled MSE; the optimizer uses the scaled errors.

Supervised sampling mixes distributions over streamwise velocity deficit, signed-log crossflow scores, and pressure. Occupied ranges receive equal proposal mass within each field, and the field distributions are combined using `SUPERVISED_SAMPLING_FIELD_WEIGHTS`. Sampling scores do not transform the physical targets. Validation points are excluded from training selection.

By default, 5% of the existing training budget is reserved for persistent near-body and wake points. The quota is split equally between those regions and across the loaded cases. Near-body points have a positive exported `wallDist` of at most 0.10 m. Wake points are farther from the body, downstream of each case's `wake_start_x`, and have a streamwise velocity deficit above 5%. Only interior mesh rows are eligible when the export includes `Block Name`. If a group falls short, its unused slots stay in the rotating sample. Validation is selected first and remains unchanged. Model input bounds and target scales use the original weighted draw. Set `SUPERVISED_PERSISTENT_FRACTION = 0` to disable the reservation or use CSVs without `wallDist`.

Wake-vorticity supervision compares the signed streamwise curl of the predicted velocity with the exported `vorticity_0` field, scaled by `l_ref / u_ref`. It samples up to 64 wake points per case per step, preferring the persistent wake subset when available. A Smooth L1 loss uses each case's fixed training RMS vorticity scale. The dashboard reports relative L2 on a separate fixed subset of validation wake points. Set `VORTICITY_LOSS_WEIGHT = 0` to disable this term. CSVs without `wallDist` need both vorticity supervision and persistent sampling disabled.

PDE collocation uses a residual-weighted candidate pool within the model-input bounding box. Collocation points are distinct from supervised and validation points.

Adam starts at `LR` and halves the learning rate every `LR_DECAY_HALF_LIFE` epochs, down to `LR_MIN`. Validation does not change the rate. Fresh boundary warm-up uses `LR`. Resuming retains Adam's optimizer state and sets the rate for the next epoch using the same schedule.

## Default configuration

| Setting | Default |
| --- | --- |
| `EPOCHS` | 500,000 |
| `PRETRAIN_EPOCHS` | 20,000 |
| `LR` | `5e-5` |
| `LR_DECAY_HALF_LIFE` | 5,000 epochs |
| `LR_MIN` | `1e-5` |
| `N_TRAIN_POINTS` | 50,000 |
| `N_VAL_POINTS` | 50,000 |
| `N_COLLOCATION` | 8,000 |
| `PDE_RAMP_EPOCHS` | 15,000 |
| `PDE_MAX_GRAD_RATIO` | 0.3 |
| `PDE_GRADIENT_EVERY` | 50 epochs |
| `SUPERVISED_REFRESH_EVERY` | 250 epochs, during pretraining |
| `SUPERVISED_PERSISTENT_FRACTION` | 0.05 |
| `SUPERVISED_NEAR_BODY_DISTANCE` | 0.10 m |
| `SUPERVISED_WAKE_DEFICIT` | 0.05 |
| `VORTICITY_LOSS_WEIGHT` | 0.05 |
| `VORTICITY_POINTS_PER_CASE` | 64 |
| `LOSS_GRADIENT_DIAGNOSTIC_EVERY` | 250 epochs, or 0 to disable |
| `COLLOCATION_REFRESH_EVERY` | 200 epochs |
| `VALIDATION_EVERY` | 20 epochs |
| `PRINT_EVERY` | 1 epoch between dashboard status writes |
| `PLOT_EVERY` | 100 epochs |
| `CHECKPOINT_SAVE_EVERY` | 100 epochs |
| `CHECKPOINT_DIR` | `checkpoints/vector_potential_boundaries` |
| `BOUNDARY_WARMUP_STEPS` | 200 |
| `BOUNDARY_POINTS_PER_GROUP` | 256 |
| `BOUNDARY_LOSS_WEIGHT` | 1.0 |
| `AUTO_RESUME` | `True` |

`PDE_MAX_GRAD_RATIO` controls the intended weighted PDE/data gradient ratio at controller updates. The learning-rate schedule continues across the transition to PDE training.

## Dashboard and checkpoints

The dashboard reads generated JSON and a combined training plot. It displays losses, fieldwise validation relative L2 errors, prediction statistics, parity plots, wake/freestream errors, PDE residual distributions, gradient diagnostics, and supervised sampling distributions.

The total-loss breakdown shows weighted scalar contributions and periodically samples weighted parameter-gradient norms before Adam. Norm shares compare magnitudes; they do not account for cancellation between gradients. Mass continuity contributes zero to the objective because the curl enforces it. Its residual is still shown separately.

| Output | Purpose |
| --- | --- |
| `dashboard/status.json` | Latest metrics and training progress |
| `dashboard/alerts.json` | Runtime alert entries |
| `dashboard/training_monitor.png` | Combined diagnostic plots |
| `checkpoints/vector_potential_boundaries/checkpoint_latest.pt` | Most recent saved training state |
| `checkpoints/vector_potential_boundaries/checkpoint_previous.pt` | Previous saved state |
| `checkpoints/vector_potential_boundaries/checkpoint_pde_start.pt` | Snapshot near the pretraining/PDE boundary |

With `AUTO_RESUME = True`, startup loads the latest checkpoint, falling back to the previous file if the latest is missing or cannot be deserialized. Checkpoints store model weights, optimizer state, loss-controller state, diagnostic histories, and the saved epoch. Training continues from the next epoch.

To start an independent run, preserve existing checkpoints in another directory and set `AUTO_RESUME = False`. Disabling resume alone does not change the output directory. Keep architecture and spatial-frequency settings compatible when resuming.

Vector-potential runs use a separate checkpoint directory to preserve earlier direct-velocity runs. Old direct-velocity checkpoints cannot be resumed with this output transformation. Loading checks the velocity representation, reference lengths, and spatial frequencies before applying weights.

Non-finite predictions, newly calculated PDE residuals, total loss, or parameter gradients raise an error before the optimizer step. Interrupting training does not trigger an extra checkpoint save; restarting uses the most recent scheduled save.

Datasets, checkpoints, generated diagnostics, dashboard JSON/images, logs, and Python caches are excluded by [.gitignore](.gitignore).

## Source layout

| File | Responsibility |
| --- | --- |
| [main.py](main.py) | Assemble the dataset, model, optimizer, diagnostics, and trainer |
| [config.py](config.py) | Case metadata, column mapping, and training settings |
| [data.py](data.py) | CSV loading, nondimensionalization, sampling, and validation split |
| [model.py](model.py) | Tanh network, spatial Fourier features, and scaled curl velocity outputs |
| [losses.py](losses.py) | Supervised loss, PDE residual, and PDE gradient controller |
| [boundaries.py](boundaries.py) | Read mesh boundary faces and apply velocity constraints |
| [colloc.py](colloc.py) | Residual-adaptive collocation sampling |
| [training.py](training.py) | Training loop, validation schedule, and learning-rate updates |
| [validation.py](validation.py) | Fieldwise and regional validation errors |
| [diagnostics.py](diagnostics.py) | Training and validation history buffers |
| [checkpoint.py](checkpoint.py) | Save, rotate, validate, and restore checkpoints |
| [dashboard/](dashboard/) | Browser interface, JSON reporting, plots, and HTTP server |
| [paraviewWallDist.txt](paraviewWallDist.txt) | Separate ParaView wall-distance utility |

## Current scope

The PDE residual implements steady incompressible Navier–Stokes with molecular viscosity. The model predicts only `u`, `v`, `w`, and `p`; exported turbulence fields such as `nut`, `k`, and `omega` are not used to close the momentum equations. `USE_FROZEN_NUT_LAPLACIAN` is currently unused. The residual therefore does not enforce a complete RANS or k–omega SST closure.

Collocation samples the axis-aligned input bounding box without excluding solid geometry. The velocity boundary loss applies soft constraints for static meshes, and time is not a model input. The PDE uses the first case's length scales and density, so cases should share those properties. Fixed validation points are drawn from the loaded cases; this split does not establish accuracy on a held-out geometry or reference velocity.
