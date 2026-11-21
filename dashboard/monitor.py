import json
import math
import time
from pathlib import Path

def _scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

class Monitor:
    def __init__(self, dashboard_dir):
        self.dashboard_dir = Path(dashboard_dir)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

        self.status_path = self.dashboard_dir / "status.json"
        self.alerts_path = self.dashboard_dir / "alerts.json"

        if not self.alerts_path.exists():
            self._write_json(self.alerts_path, {"alerts": []})

    def _write_json(self, path, obj):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(_json_safe(obj), f, indent=2, default=str, allow_nan=False)
        tmp.replace(path)

    def alert(self, message, severity="info", epoch=None):
        try:
            with open(self.alerts_path) as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {"alerts": []}

        payload["alerts"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "severity": severity,
            "message": message,
        })
        payload["alerts"] = payload["alerts"][-200:]
        self._write_json(self.alerts_path, payload)

    def update(self, trainer, plot_updated=False):
        from diagnostics import LogCol, ValLogCol

        diag = trainer.diagnostics
        lc = trainer.loss_controller

        row = diag.trainingHistory[diag._log_idx - 1].tolist() if diag._log_idx > 0 else None
        val_row = diag.validationHistory[diag._val_idx - 1].tolist() if diag._val_idx > 0 else None

        elapsed_min = (time.time() - self._start_time) / 60.0
        progress = trainer.epoch / trainer.config.EPOCHS
        eta_min = (elapsed_min / progress - elapsed_min) if progress > 0 else None

        status = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Training" if trainer.epoch < trainer.config.EPOCHS else "Complete",
            "phase": {-1: "Warmup", 0: "Supervised", 1: "PDE ramp"}.get(lc.status, "?"),
            "epoch": trainer.epoch,
            "total_epochs": trainer.config.EPOCHS,
            "progress_percent": round(100 * progress, 3),
            "status_update_every": trainer.config.PRINT_EVERY,
            "image_update_every": trainer.config.PLOT_EVERY,
            "plot_epoch": trainer.epoch - (trainer.epoch % trainer.config.PLOT_EVERY),
            "plot_updated": int(plot_updated),
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "time": {
                "elapsed_minutes": elapsed_min,
                "eta_minutes": eta_min,
            },

            "losses": None,
            "field_losses": None,
            "validation": None,
            "pde": None,

            "statistics": None,
            "residuals": None,
            "pde_stability": None,
        }

        if row is not None:
            data_norm, pde_norm = lc.data_grad_norm, lc.pde_grad_norm
            raw_grad_ratio = pde_norm/data_norm if data_norm > 0 else None
            actual_grad_ratio = lc.lambda_pde*raw_grad_ratio if raw_grad_ratio is not None else None
            status["losses"] = {
                "total": lc.latest_total_loss,
                "data": lc.latest_data_loss,
                "pde_raw": lc.latest_pde_loss,
                "weighted_pde": lc.latest_weighted_pde_loss,
                "boundary_raw": lc.latest_boundary_loss,
                "weighted_boundary": lc.latest_weighted_boundary_loss,
                "vorticity_raw": lc.latest_vorticity_loss,
                "weighted_vorticity": lc.latest_weighted_vorticity_loss,
            }
            status["objective"] = {
                "weighted_terms": {
                    "data": lc.latest_data_loss,
                    "boundary": lc.latest_weighted_boundary_loss,
                    "vorticity": lc.latest_weighted_vorticity_loss,
                    "momentum": lc.latest_weighted_pde_loss,
                    "mass": 0.0,
                },
                "gradients": lc.latest_objective_gradients,
            }
            status["field_losses"] = {"u": row[LogCol.U_LOSS], "v": row[LogCol.V_LOSS], "w": row[LogCol.W_LOSS], "p": row[LogCol.P_LOSS]}
            status["pde"] = {
                "effective_lambda": row[LogCol.LAMBDA_PDE],
                "cov": row[LogCol.COV], "mean": row[LogCol.MEAN],
                "std": row[LogCol.STD], "slope": row[LogCol.SLOPE],
                "gradient_cosine": lc.gradient_cosine,
                "data_grad_norm": lc.data_grad_norm,
                "pde_grad_norm": lc.pde_grad_norm,
                "weighted_pde_grad_norm": lc.lambda_pde*lc.pde_grad_norm,
                "target_grad_ratio": lc.pde_grad_ratio,
                "desired_grad_ratio": lc.pde_desired_ratio,
                "scheduled_ratio_cap": lc.pde_ratio_cap,
                "cosine_gate": lc.pde_cosine_gate,
                "actual_grad_ratio": actual_grad_ratio,
            }
            status["comparisons"] = {
                "weighted_pde_to_data": lc.latest_weighted_pde_loss/lc.latest_data_loss if lc.latest_data_loss > 0 else None,
                "pde_to_data_grad_ratio": raw_grad_ratio,
                "actual_weighted_grad_ratio": actual_grad_ratio,
            }
            status["pde_stability"] = {
                "mean": row[LogCol.MEAN],
                "std": row[LogCol.STD],
                "cov": row[LogCol.COV],
                "slope": row[LogCol.SLOPE],
                "window": lc.window,
            }
            
        pde_components = getattr(trainer,"latest_pde_components",None,)

        if pde_components is not None:
            status["residuals"] = {
                "component": {
                    "rx": _scalar(pde_components.get("mom_x")),
                    "ry": _scalar(pde_components.get("mom_y")),
                    "rz": _scalar(pde_components.get("mom_z")),
                    "rc": _scalar(pde_components.get("continuity")),
                },
                "rms": {
                    "rx": _scalar(pde_components.get("rx_rms")),
                    "ry": _scalar(pde_components.get("ry_rms")),
                    "rz": _scalar(pde_components.get("rz_rms")),
                    "rc": _scalar(pde_components.get("rc_rms")),
                },
                "max": {
                    "rx": _scalar(pde_components.get("rx_max")),
                    "ry": _scalar(pde_components.get("ry_max")),
                    "rz": _scalar(pde_components.get("rz_max")),
                    "rc": _scalar(pde_components.get("rc_max")),
                },
            }

        if val_row is not None:
            status["validation"] = {
                "u": val_row[ValLogCol.L2_U],
                "v": val_row[ValLogCol.L2_V],
                "w": val_row[ValLogCol.L2_W],
                "p": val_row[ValLogCol.L2_P],
                "wake_error": val_row[ValLogCol.WAKE_ERROR],
                "freestream_error": val_row[ValLogCol.FREESTREAM_ERROR],
                "overall_error": val_row[ValLogCol.OVERALL_ERROR],
            }
            status["vorticity_validation"] = getattr(trainer, "latest_vorticity_validation", None)
        
        latest_stats = getattr(trainer, "latest_stats", None)

        if latest_stats is not None:
            status["statistics"] = latest_stats

        status["last_checkpoint"] = getattr(trainer.checkpoint_manager, "last_save_info", {"epoch": None, "status": "No checkpoint saved yet"})
        status["save_log"] = getattr(trainer.checkpoint_manager, "save_log", [])[-50:]

        self._write_json(self.status_path, status)
        
