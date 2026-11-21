import torch
import torch.nn as nn
from pathlib import Path
from config import params
from model import PINN
from data import OpenFOAMDataSet
from losses import LossController, navier_stokes_residual, pde_loss
from training import Trainer
from diagnostics import Diagnoser
from colloc import Collocator
from checkpoint import CheckpointManager
from boundaries import VelocityBoundaries
from dashboard.monitor import Monitor

def main():
    config = params()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_files = [case["path"] for case in config.CASE_FILES]
    data = OpenFOAMDataSet(
        csv_files, ["x","y","z","u_ref"], ["u","v","w","p"], config.N_TRAIN_POINTS, config.N_VAL_POINTS,
        seed=config.SEED, persistent_fraction=config.SUPERVISED_PERSISTENT_FRACTION,
        near_body_distance=config.SUPERVISED_NEAR_BODY_DISTANCE, wake_deficit=config.SUPERVISED_WAKE_DEFICIT,
        supervise_vorticity=config.VORTICITY_LOSS_WEIGHT > 0,
    )
    lower, upper = data.training_bounds
    l_ref, cross_ref = config.CASE_FILES[0]["l_ref"], config.CASE_FILES[0]["cross_ref"]
    if any((case["l_ref"], case["cross_ref"]) != (l_ref, cross_ref) for case in config.CASE_FILES):
        raise ValueError("Coupled velocity outputs require shared reference lengths across cases")
    model = PINN(config.INPUT_VECTOR, config.OUTPUT_VECTOR, config.HIDDEN_WIDTH, config.HIDDEN_LAYERS, lower, upper,
                 spatial_frequencies=config.SPATIAL_FOURIER_FREQUENCIES, l_ref=l_ref, cross_ref=cross_ref).to(device)
    with torch.no_grad():
        # Constant potential offsets do not change velocity; initialize only pressure.
        model.output_layer.bias[:3].zero_()
        model.output_layer.bias[3].copy_(data.target_mean[3].to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    scheduler = None
    
    data_loss = nn.MSELoss()
    
    loss_controller = LossController(u_loss_fn = data_loss, v_loss_fn = data_loss, w_loss_fn = data_loss, p_loss_fn = data_loss, pde_loss_fn = pde_loss)
    
    collocator = Collocator(lower, upper, model, pde_residual = navier_stokes_residual, pool=20000, refresh = config.COLLOCATION_REFRESH_EVERY, device = device)
    
    diagnostics = Diagnoser(loss_controller=loss_controller, u_ref=max(case["u_ref"] for case in config.CASE_FILES), config = config, model = model, data = data, save_dir = "diagnostics")

    
    monitor = Monitor("dashboard")
    
    checkpoint_manager = CheckpointManager(config.CHECKPOINT_DIR)
    boundaries = None
    if config.BOUNDARY_WARMUP_STEPS > 0 or config.BOUNDARY_LOSS_WEIGHT > 0:
        boundaries = VelocityBoundaries(config.CASE_FILES, config.BOUNDARY_POINTS_PER_GROUP,
                                        config.SEED, device)

    trainer = Trainer(model, dashboard_monitor=monitor, optimizer=optimizer, data=data, pde_residual=navier_stokes_residual, loss_controller=loss_controller, config=config, diagnostics=diagnostics, collocator=collocator, checkpoint_manager=checkpoint_manager, scheduler=scheduler, device=device, boundaries=boundaries)
    
    if (config.AUTO_RESUME and checkpoint_manager.has_checkpoint()):
        checkpoint_manager.load(trainer,which="latest")
        interval = config.SUPERVISED_REFRESH_EVERY
        refresh_epoch = min(trainer.epoch - 1, config.PRETRAIN_EPOCHS) // interval * interval
        if refresh_epoch > 0:
            data.refresh_training(config.SEED + refresh_epoch)
    else:
        trainer.warmup_boundaries()
        
    trainer.train()

if __name__ == "__main__":
    main()
