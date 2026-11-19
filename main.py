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
from dashboard.monitor import Monitor

def main():
    config = params()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_files = [case["path"] for case in config.CASE_FILES]
    data = OpenFOAMDataSet(csv_files, ["x","y","z","u_ref"],["u","v","w","p"],config.N_TRAIN_POINTS, config.N_VAL_POINTS)
    lower, upper = data.train_inputs.min(dim=0).values, data.train_inputs.max(dim=0).values
    model = PINN(config.INPUT_VECTOR, config.OUTPUT_VECTOR, config.HIDDEN_WIDTH, config.HIDDEN_LAYERS, lower, upper,
                 spatial_frequencies=config.SPATIAL_FOURIER_FREQUENCIES).to(device)
    with torch.no_grad():
        model.output_layer.bias.copy_(data.train_targets.mean(dim=0).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    scheduler = None
    
    data_loss = nn.MSELoss()
    
    loss_controller = LossController(u_loss_fn = data_loss, v_loss_fn = data_loss, w_loss_fn = data_loss, p_loss_fn = data_loss, pde_loss_fn = pde_loss)
    
    collocator = Collocator(lower, upper, model, pde_residual = navier_stokes_residual, pool=20000, refresh = config.COLLOCATION_REFRESH_EVERY, device = device)
    
    diagnostics = Diagnoser(loss_controller=loss_controller, u_ref=max(case["u_ref"] for case in config.CASE_FILES), config = config, model = model, data = data, save_dir = "diagnostics")

    
    monitor = Monitor("dashboard")
    
    checkpoint_manager = CheckpointManager("checkpoints")

    trainer = Trainer(model, dashboard_monitor=monitor, optimizer=optimizer, data=data, pde_residual=navier_stokes_residual, loss_controller=loss_controller, config=config, diagnostics=diagnostics, collocator=collocator, checkpoint_manager=checkpoint_manager, scheduler=scheduler, device=device)
    
    if (config.AUTO_RESUME and checkpoint_manager.has_checkpoint()):
        checkpoint_manager.load(trainer,which="latest")
        
    trainer.train()

if __name__ == "__main__":
    main()
