import torch
from pathlib import Path

class LogCol:
    EPOCH = 0
    U_LOSS = 1
    V_LOSS = 2
    W_LOSS = 3
    P_LOSS = 4
    PDE_LOSS = 5
    LAMBDA_U = 6
    LAMBDA_V = 7
    LAMBDA_W = 8
    LAMBDA_P = 9
    LAMBDA_PDE = 10
    TOTAL_LOSS = 11
    COV = 12
    MEAN = 13
    STD = 14
    SLOPE = 15
    MOM_X = 16
    MOM_Y = 17
    MOM_Z = 18
    CONT = 19
    GRAD_COSINE = 20
    DATA_GRAD_NORM = 21
    PDE_GRAD_NORM = 22
    TARGET_GRAD_RATIO = 23
    WEIGHTED_BOUNDARY_LOSS = 24
    WEIGHTED_VORTICITY_LOSS = 25

class ValLogCol:
    EPOCH = 0
    L2_U = 1
    L2_V = 2
    L2_W = 3
    L2_P = 4
    WAKE_ERROR = 5
    FREESTREAM_ERROR = 6
    OVERALL_ERROR = 7


class Diagnoser:
    def __init__(self, loss_controller, u_ref, config, model, data, save_dir=None):
        self.loss_controller = loss_controller
        self.u_ref = u_ref
        self.config = config
        self.model = model
        self.data = data
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.trainingHistory = torch.zeros(config.EPOCHS, 26)
        self._log_idx = 0

        last_epoch = config.EPOCHS - 1
        num_val_points = last_epoch // config.VALIDATION_EVERY + 1
        self.validationHistory = torch.zeros(num_val_points, 8)
        self._val_idx = 0

        self.latest_val_extras = None

    def log(self, epoch, loss, pde_components):
        cov, mean, std = self.loss_controller.pde_stability(
            self.loss_controller.loss_history, self.loss_controller.window
        )
        slope = self.loss_controller.pde_trend(
            self.loss_controller.loss_history, self.loss_controller.window
        )
        mom_x = float(pde_components["rx_rms"].detach().cpu())
        mom_y = float(pde_components["ry_rms"].detach().cpu())
        mom_z = float(pde_components["rz_rms"].detach().cpu())
        cont = float(pde_components["rc_rms"].detach().cpu())
        logRow = torch.hstack([
            torch.tensor([epoch], dtype=torch.float32),
            torch.tensor(loss, dtype=torch.float32),
            torch.tensor([cov, mean, std, slope], dtype=torch.float32),
            torch.tensor([mom_x, mom_y, mom_z, cont], dtype = torch.float32)
            ,torch.tensor([
                self.loss_controller.gradient_cosine,
                self.loss_controller.data_grad_norm,
                self.loss_controller.pde_grad_norm,
                self.loss_controller.pde_grad_ratio,
                self.loss_controller.latest_weighted_boundary_loss,
                self.loss_controller.latest_weighted_vorticity_loss,
            ], dtype=torch.float32)
        ])
        self.trainingHistory[self._log_idx] = logRow
        self._log_idx += 1

    def log_validation(self, row, extras):
        self.validationHistory[self._val_idx] = row
        self._val_idx += 1
        self.latest_val_extras = extras

    def _filled(self):
        return self.trainingHistory[:self._log_idx].numpy()

    def _filled_val(self):
        return self.validationHistory[:self._val_idx].numpy()
    
    
