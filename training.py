from enum import Enum, auto
import torch
from config import params
from losses import LossController
from validation import validation
from dashboard.monitor import Monitor
from dashboard.plotting import render_training_monitor, save_parity_plots
from dashboard.server import start_dashboard_server
import webbrowser

class TrainingPhase(Enum):
    SUPERVISED = auto()
    PDE_AUTO = auto()


class Trainer:
    def __init__(self, model, dashboard_monitor, optimizer, data, pde_residual, loss_controller, config, diagnostics, collocator, checkpoint_manager, scheduler, device):
        self.model = model
        self.dashboard_monitor = dashboard_monitor
        self.optimizer = optimizer
        self.data = data
        self.pde_residual = pde_residual
        self.loss_controller = loss_controller
        self.config = config
        self.diagnostics = diagnostics
        self.latest_metrics = None
        self.latest_stats = None
        self.latest_fields = None
        self.latest_pde_components = None
        self.collocator = collocator
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.plotNow = False
        self.saveNow = False
        self.latest_colloc = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_lr = float(config.LR)

        self.model.to(self.device)

        self.target_scales = (self.data.train_targets.std(dim=0).to(self.device).clamp_min(1e-6))
        
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
        if self.epoch > 0 and self.diagnostics._val_idx > 0:
            last_validation = self.diagnostics.validationHistory[self.diagnostics._val_idx - 1]
            self.update_learning_rate(last_validation[1:5])
        self.dashboard_server = start_dashboard_server(self.dashboard_monitor.dashboard_dir,port=8000)
        webbrowser.open("http://localhost:8000/index.html")
        while self.epoch < self.config.EPOCHS:
            toLog = self.trainStep()
            self.diagnostics.log(self.epoch,toLog,self.latest_pde_components)

            self.plotNow = (self.epoch % self.config.PLOT_EVERY == 0)
            self.saveNow = (self.epoch % self.config.CHECKPOINT_SAVE_EVERY == 0)

            update_lr = self.epoch > 0 and self.epoch % self.config.LR_UPDATE_EVERY == 0
            if self.epoch % self.config.VALIDATION_EVERY == 0 or update_lr:
                self.validate()
            if update_lr:
                self.update_learning_rate([self.latest_metrics[name] for name in ("u", "v", "w", "p")])
            if self.plotNow:
                render_training_monitor(
                    self.diagnostics,
                    self.dashboard_monitor.dashboard_dir / "training_monitor.png",
                    self._last_residual, 
                    model = self.model, 
                    x_colloc = self.latest_colloc, 
                    device = self.device, 
                    latest_fields = self.latest_fields, 
                    latest_metrics = self.latest_metrics, 
                    monitor = self.dashboard_monitor.dashboard_dir,
                )
            if self.epoch % self.config.PRINT_EVERY == 0:
                self.dashboard_monitor.update(self, plot_updated=self.plotNow)

            if self.saveNow:
                self.checkpoint_manager.save(self, self.epoch)
                self.saveNow = False

            self.epoch += 1

    def update_learning_rate(self, l2_errors):
        errors = torch.as_tensor(l2_errors, dtype=torch.float64)
        if errors.shape != (4,):
            raise ValueError("Expected four validation L2 errors in [u, v, w, p] order")
        self._require_finite(errors, "validation L2 errors for learning-rate update")
        if (errors < 0).any():
            raise ValueError("Validation L2 errors must be nonnegative")
        average_l2 = errors.mean().item()
        learning_rate = self.initial_lr * average_l2
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        print(f"LR update at epoch {self.epoch}: average validation L2 = {100*average_l2:.2f}%, LR = {learning_rate:.6g}")

    def trainStep(self):
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
            self.latest_colloc = x_colloc.detach()
            x_colloc.requires_grad_(True)
            residual = self.pde_residual(self.model, x_colloc)
            self._require_finite(residual, "PDE residual")
            self._last_residual = residual.detach()
        else:
            residual = self._last_residual

        self.latest_pde_components = {
            "mom_x": residual[:, 0].square().mean(),
            "mom_y": residual[:, 1].square().mean(),
            "mom_z": residual[:, 2].square().mean(),
            "continuity": residual[:, 3].square().mean(),

            "rx_rms": residual[:, 0].square().mean().sqrt(),
            "ry_rms": residual[:, 1].square().mean().sqrt(),
            "rz_rms": residual[:, 2].square().mean().sqrt(),
            "rc_rms": residual[:, 3].square().mean().sqrt(),

            "rx_max": residual[:, 0].abs().max(),
            "ry_max": residual[:, 1].abs().max(),
            "rz_max": residual[:, 2].abs().max(),
            "rc_max": residual[:, 3].abs().max(),
        }

        loss_residual = residual.detach() if supervised else residual
        loss = self.loss_controller.update(pred, targets, loss_residual, epoch=self.epoch, model=self.model)
        self._require_finite(loss, "total loss")
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self._require_finite(parameter.grad, f"gradient for {name}")
        self.optimizer.step()
        return self.loss_controller.report_LL()

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
        self.model.train()
        self.diagnostics.log_validation(row, extras)
        self.latest_metrics = {
            "u": float(row[1]),
            "v": float(row[2]),
            "w": float(row[3]),
            "p": float(row[4]),
        }
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
