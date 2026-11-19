import torch
from pathlib import Path


class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._pde_checkpoint_saved = False

    def _latest_path(self):
        return self.save_dir / "checkpoint_latest.pt"

    def _previous_path(self):
        return self.save_dir / "checkpoint_previous.pt"

    def _pde_start_path(self):
        return self.save_dir / "checkpoint_pde_start.pt"

    def _build_state(self, trainer, epoch):
        return {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "model_features": {"spatial_frequencies": trainer.model.spatial_frequencies},
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,

            "loss_controller": {
                "lambda_u": trainer.loss_controller.lambda_u,
                "lambda_v": trainer.loss_controller.lambda_v,
                "lambda_w": trainer.loss_controller.lambda_w,
                "lambda_p": trainer.loss_controller.lambda_p,
                "lambda_pde": trainer.loss_controller.lambda_pde,
                "status": trainer.loss_controller.status,
                "loss_history": trainer.loss_controller.loss_history,
                "pde_grad_ratio": trainer.loss_controller.pde_grad_ratio,
                "gradient_cosine": trainer.loss_controller.gradient_cosine,
                "data_grad_norm": trainer.loss_controller.data_grad_norm,
                "pde_grad_norm": trainer.loss_controller.pde_grad_norm,
                "pde_ratio_cap": trainer.loss_controller.pde_ratio_cap,
                "pde_cosine_gate": trainer.loss_controller.pde_cosine_gate,
                "pde_desired_ratio": trainer.loss_controller.pde_desired_ratio,
            },

            "diagnostics": {
                "trainingHistory": trainer.diagnostics.trainingHistory,
                "_log_idx": trainer.diagnostics._log_idx,
                "validationHistory": trainer.diagnostics.validationHistory,
                "_val_idx": trainer.diagnostics._val_idx,
            },

            "phase": trainer.phase,
        }

    def save(self, trainer, epoch):
        state = self._build_state(trainer, epoch)

        latest, previous = self._latest_path(), self._previous_path()
        temporary = self.save_dir / "checkpoint_latest.tmp"
        torch.save(state, temporary)
        if latest.exists():
            latest.replace(previous)
        temporary.replace(latest)

        if not self._pde_checkpoint_saved and epoch >= trainer.config.PRETRAIN_EPOCHS:
            torch.save(state, self._pde_start_path())
            self._pde_checkpoint_saved = True

    def load(self, trainer, which="latest"):
        path = self._latest_path() if which == "latest" else self._pde_start_path()
        if not path.exists():
            if which == "latest" and self._previous_path().exists():
                path = self._previous_path()
            else:
                raise FileNotFoundError(f"No checkpoint found at {path}")

        try:
            state = torch.load(path, map_location=trainer.device, weights_only=False)
        except Exception:
            if which != "latest" or path == self._previous_path() or not self._previous_path().exists():
                raise
            path = self._previous_path()
            print(f"Latest checkpoint could not be loaded; falling back to {path}")
            state = torch.load(path, map_location=trainer.device, weights_only=False)

        self._prepare_spatial_features(trainer, state)
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        if trainer.scheduler is not None and state["scheduler_state_dict"] is not None:
            trainer.scheduler.load_state_dict(state["scheduler_state_dict"])

        lc = state["loss_controller"]
        trainer.loss_controller.lambda_u = lc["lambda_u"]
        trainer.loss_controller.lambda_v = lc["lambda_v"]
        trainer.loss_controller.lambda_w = lc["lambda_w"]
        trainer.loss_controller.lambda_p = lc["lambda_p"]
        trainer.loss_controller.lambda_pde = lc["lambda_pde"]
        trainer.loss_controller.status = lc["status"]
        trainer.loss_controller.loss_history = lc["loss_history"]
        for name in ("pde_grad_ratio", "gradient_cosine", "data_grad_norm", "pde_grad_norm",
                     "pde_ratio_cap", "pde_cosine_gate", "pde_desired_ratio"):
            if name in lc:
                setattr(trainer.loss_controller, name, lc[name])

        diag = state["diagnostics"]
        old_history = diag["trainingHistory"]
        history = trainer.diagnostics.trainingHistory
        rows, cols = min(old_history.shape[0], history.shape[0]), min(old_history.shape[1], history.shape[1])
        history[:rows, :cols] = old_history[:rows, :cols]
        trainer.diagnostics._log_idx = min(diag["_log_idx"], history.shape[0])

        old_validation = diag["validationHistory"]
        validation = trainer.diagnostics.validationHistory
        rows, cols = min(old_validation.shape[0], validation.shape[0]), min(old_validation.shape[1], validation.shape[1])
        validation[:rows, :cols] = old_validation[:rows, :cols]
        trainer.diagnostics._val_idx = min(diag["_val_idx"], validation.shape[0])

        trainer.epoch = state["epoch"]+1
        trainer.phase = state["phase"]

    @staticmethod
    def _prepare_spatial_features(trainer, state):
        saved_features = state.get("model_features")
        if saved_features is not None:
            if tuple(saved_features["spatial_frequencies"]) != trainer.model.spatial_frequencies:
                raise ValueError("Checkpoint spatial frequencies differ from config; restore the checkpoint's frequency setting before resuming.")
            return

        old_weight = state["model_state_dict"]["input_layer.weight"]
        new_weight = trainer.model.input_layer.weight
        if old_weight.shape == new_weight.shape:
            return
        if (old_weight.shape[0] != new_weight.shape[0]
                or old_weight.shape[1] != trainer.model.input_min.numel()
                or old_weight.shape[1] >= new_weight.shape[1]):
            raise ValueError("Checkpoint input layer is incompatible with the configured model.")

        padding = new_weight.shape[1] - old_weight.shape[1]
        state["model_state_dict"]["input_layer.weight"] = torch.nn.functional.pad(old_weight, (0, padding))
        optimizer = state["optimizer_state_dict"]
        for saved_group, current_group in zip(optimizer["param_groups"], trainer.optimizer.param_groups):
            for saved_id, parameter in zip(saved_group["params"], current_group["params"]):
                if parameter is new_weight:
                    moments = optimizer["state"].get(saved_id, {})
                    for name, value in list(moments.items()):
                        if isinstance(value, torch.Tensor) and value.shape == old_weight.shape:
                            moments[name] = torch.nn.functional.pad(value, (0, padding))
        print(f"Migrated checkpoint: added {padding} spatial features with zero initial weights; learned outputs preserved.")

    def has_checkpoint(self, which="latest"):
        if which == "latest":
            return self._latest_path().exists() or self._previous_path().exists()
        return self._pde_start_path().exists()
