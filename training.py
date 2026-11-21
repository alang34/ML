from enum import Enum, auto
import torch
from torch.nn.functional import smooth_l1_loss
from losses import streamwise_vorticity
from validation import validation
from dashboard.plotting import render_training_monitor
from dashboard.server import start_dashboard_server
import webbrowser

class TrainingPhase(Enum):
    SUPERVISED = auto()
    PDE_AUTO = auto()


class Trainer:
    def __init__(self, model, dashboard_monitor, optimizer, data, pde_residual, loss_controller, config, diagnostics, collocator, checkpoint_manager, scheduler, device, boundaries=None):
        self.model = model
        self.dashboard_monitor = dashboard_monitor
        self.optimizer = optimizer
        self.data = data
        self.pde_residual = pde_residual
        self.loss_controller = loss_controller
        self.boundaries = boundaries
        self.config = config
        self.diagnostics = diagnostics
        self.latest_stats = None
        self.latest_fields = None
        self.latest_pde_components = None
        self.latest_vorticity_validation = None
        self.collocator = collocator
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.plotNow = False
        self.saveNow = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_lr = float(config.LR)
        if getattr(config, "VORTICITY_LOSS_WEIGHT", 0.0) > 0 and config.VORTICITY_POINTS_PER_CASE <= 0:
            raise ValueError("Wake vorticity supervision requires VORTICITY_POINTS_PER_CASE > 0")
        if config.LR_DECAY_HALF_LIFE <= 0 or not 0 < config.LR_MIN <= self.initial_lr:
            raise ValueError("Learning rate decay requires a positive half-life and 0 < LR_MIN <= LR")

        self.model.to(self.device)

        target_scales = getattr(self.data, "target_scales", None)
        if target_scales is None:
            target_scales = self.data.train_targets.std(dim=0).clamp_min(1e-6)
        self.target_scales = target_scales.to(self.device)
        
        if self.target_scales.shape != (4,):
            raise ValueError(
                "Expected four supervised target scales in [u, v, w, p] order, "
                f"got shape {tuple(self.target_scales.shape)}"
            )
        self.loss_controller.set_target_scales(self.target_scales)
        print("Supervised target scales [u, v, w, p]:", self.target_scales)

        self.epoch = 0
        self.phase = TrainingPhase.SUPERVISED
        self._last_residual = None

    def _require_finite(self, value, name):
        if not torch.isfinite(value).all():
            bad = (~torch.isfinite(value)).sum().item()
            raise FloatingPointError(f"Non-finite {name} at epoch {self.epoch} ({bad}/{value.numel()} values)")

    def train(self):
        self.dashboard_server = start_dashboard_server(self.dashboard_monitor.dashboard_dir,port=8000)
        webbrowser.open("http://localhost:8000/index.html")
        while self.epoch < self.config.EPOCHS:
            toLog = self.trainStep()
            self.diagnostics.log(self.epoch,toLog,self.latest_pde_components)

            self.plotNow = (self.epoch % self.config.PLOT_EVERY == 0)
            self.saveNow = (self.epoch % self.config.CHECKPOINT_SAVE_EVERY == 0)

            validation_due = self.epoch % self.config.VALIDATION_EVERY == 0
            if validation_due:
                self.validate()
            if self.plotNow and self.latest_fields is not None:
                render_training_monitor(
                    self.diagnostics,
                    self.dashboard_monitor.dashboard_dir / "training_monitor.png",
                    self._last_residual,
                    self.latest_fields,
                )
            else:
                self.plotNow = False
            if self.epoch % self.config.PRINT_EVERY == 0:
                self.dashboard_monitor.update(self, plot_updated=self.plotNow)

            if self.saveNow:
                self.checkpoint_manager.save(self, self.epoch)
                self.saveNow = False

            self.epoch += 1

    def warmup_boundaries(self):
        if self.boundaries is None or self.config.BOUNDARY_WARMUP_STEPS == 0:
            return
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        evaluation_batch = self.boundaries.sample()
        with torch.no_grad():
            initial = self.boundaries.loss(self.model, evaluation_batch).item()
        print(f"Boundary initialization: {self.config.BOUNDARY_WARMUP_STEPS} steps, initial loss {initial:.6g}", flush=True)
        for _ in range(self.config.BOUNDARY_WARMUP_STEPS):
            optimizer.zero_grad()
            loss = self.boundaries.loss(self.model, self.boundaries.sample())
            self._require_finite(loss, "boundary warm-up loss")
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            final = self.boundaries.loss(self.model, evaluation_batch)
            self._require_finite(final, "boundary warm-up evaluation")
        print(f"Boundary initialization complete: loss {final.item():.6g}", flush=True)

    def trainStep(self):
        self.update_learning_rate()
        self.optimizer.zero_grad()
        if (
            0 < self.epoch <= self.config.PRETRAIN_EPOCHS
            and self.epoch % self.config.SUPERVISED_REFRESH_EVERY == 0
        ):
            self.data.refresh_training(self.config.SEED + self.epoch)
            print(f"Refreshed supervised training set at epoch {self.epoch}")
        x_data = self.data.train_inputs.to(self.device)
        targets = self.data.train_targets.to(self.device)
        pred = self.model(x_data)
        self._require_finite(pred, "model prediction")

        supervised = self.epoch <= self.config.PRETRAIN_EPOCHS
        check_pde = (
            not supervised
            or self._last_residual is None
            or self.epoch % self.config.PDE_DIAGNOSTIC_EVERY == 0
        )
        if check_pde:
            x_colloc = self.collocator.getRefresh(self.config.N_COLLOCATION, epoch=self.epoch).to(self.device)
            x_colloc.requires_grad_(True)
            residual = self.pde_residual(self.model, x_colloc)
            self._require_finite(residual, "PDE residual")
            self._last_residual = residual.detach()
        if supervised:
            residual = self._last_residual

        diagnostics_residual = residual.detach()
        self.latest_pde_components = {
            "mom_x": diagnostics_residual[:, 0].square().mean(),
            "mom_y": diagnostics_residual[:, 1].square().mean(),
            "mom_z": diagnostics_residual[:, 2].square().mean(),
            "continuity": diagnostics_residual[:, 3].square().mean(),

            "rx_rms": diagnostics_residual[:, 0].square().mean().sqrt(),
            "ry_rms": diagnostics_residual[:, 1].square().mean().sqrt(),
            "rz_rms": diagnostics_residual[:, 2].square().mean().sqrt(),
            "rc_rms": diagnostics_residual[:, 3].square().mean().sqrt(),

            "rx_max": diagnostics_residual[:, 0].abs().max(),
            "ry_max": diagnostics_residual[:, 1].abs().max(),
            "rz_max": diagnostics_residual[:, 2].abs().max(),
            "rc_max": diagnostics_residual[:, 3].abs().max(),
        }

        boundary_loss = None
        if self.boundaries is not None and self.config.BOUNDARY_LOSS_WEIGHT > 0:
            boundary_loss = self.boundaries.loss(self.model, self.boundaries.sample())
        vorticity_weight = getattr(self.config, "VORTICITY_LOSS_WEIGHT", 0.0)
        vorticity_loss = self.wake_vorticity_loss() if vorticity_weight > 0 else None
        gradient_interval = getattr(self.config, "LOSS_GRADIENT_DIAGNOSTIC_EVERY", 0)
        loss = self.loss_controller.update(
            pred, targets, residual, epoch=self.epoch, model=self.model,
            boundary_loss=boundary_loss, boundary_weight=getattr(self.config, "BOUNDARY_LOSS_WEIGHT", 0.0),
            vorticity_loss=vorticity_loss, vorticity_weight=vorticity_weight,
            measure_gradients=gradient_interval > 0 and (
                self.loss_controller.latest_objective_gradients is None or self.epoch % gradient_interval == 0
            ),
        )
        self._require_finite(loss, "total loss")
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self._require_finite(parameter.grad, f"gradient for {name}")
        self.optimizer.step()
        return self.loss_controller.report_LL()

    def wake_vorticity_loss(self):
        count = self.config.VORTICITY_POINTS_PER_CASE
        generator = torch.Generator().manual_seed(self.config.SEED + self.epoch)
        inputs, targets, scales = [], [], []
        for points, labels, scale in self.data.vorticity_training:
            idx = torch.randperm(len(points), generator=generator)[:count]
            inputs.append(points[idx])
            targets.append(labels[idx])
            scales.append(scale.expand(len(idx)))
        prediction = streamwise_vorticity(self.model, torch.cat(inputs).to(self.device), create_graph=True)
        error = (prediction - torch.cat(targets).to(self.device)) / torch.cat(scales).to(self.device)
        return smooth_l1_loss(error, torch.zeros_like(error))

    def update_learning_rate(self):
        # Use the absolute epoch so a restart continues the same decay schedule.
        learning_rate = max(
            self.config.LR_MIN,
            self.initial_lr * 0.5 ** (self.epoch / self.config.LR_DECAY_HALF_LIFE),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            row, extras = validation(
                self.model,
                self.data.val_inputs.to(self.device),
                self.data.val_targets.to(self.device),
                self.data.val_deficit.to(self.device),
                epoch=self.epoch,
            )
            wake = getattr(self.data, "vorticity_validation", ())
            if wake:
                points, targets = map(torch.cat, zip(*wake))
                prediction = streamwise_vorticity(self.model, points.to(self.device), create_graph=False).cpu()
                error = prediction - targets
                self.latest_vorticity_validation = {
                    "relative_l2": float(error.norm() / targets.norm().clamp_min(1e-12)),
                    "rms": float(error.square().mean().sqrt()),
                    "count": len(targets),
                }
        self.model.train()
        self.diagnostics.log_validation(row, extras)
        pred = extras["pred"]
        target = extras["target"]
        field_names = ["u", "v", "w", "p"]
        self.latest_stats = {}
        for i, name in enumerate(field_names):
            pred_i = pred[:, i]
            true_i = target[:, i]

            true_std = true_i.std(unbiased=True)

            self.latest_stats[name] = {
                "pred_mean": float(pred_i.mean()),
                "true_mean": float(true_i.mean()),
                "pred_std": float(pred_i.std(unbiased=True)),
                "true_std": float(true_std),
                "std_ratio": float(
                    pred_i.std(unbiased=True)
                    / true_std.clamp_min(1e-12)
                ),
            }

        self.latest_fields = {
            name: (
                pred[:, i],
                target[:, i],
            )
            for i, name in enumerate(field_names)
        }

        return row, extras
