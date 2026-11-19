from pathlib import Path

class params:
    SCRIPT_DIR = Path(__file__).resolve().parent
    CASE_FILES = [
        {
            "path": SCRIPT_DIR / "Data/Case1/33.csv",
            "u_ref": 33.27,
            "l_ref": 3.83,
            "cross_ref":48.7,
            "rho_ref": 0.8499,
            "case_id": "speed_33_27",
        },
        {
            "path": SCRIPT_DIR / "Data/Case2/83.csv",
            "u_ref": 82.6,
            "l_ref": 3.83,
            "cross_ref":48.7,
            "rho_ref": 0.8499,
            "case_id": "speed_82_6",
        },
    ]

    EPOCHS = 500000
    PRETRAIN_EPOCHS = 20000

    LR = 1e-4
    LR_UPDATE_EVERY = 50
    PDE_LR = 5e-6
    PDE_RAMP_EPOCHS = 15000
    PDE_MAX_GRAD_RATIO = 0.3
    PDE_GRADIENT_EVERY = 50
    PDE_LAMBDA_MAX_GROWTH_FACTOR = 2.0

    N_TRAIN_POINTS = 50000
    N_VAL_POINTS = 50000
    SUPERVISED_SAMPLING_FIELD_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

    N_COLLOCATION = 8000
    PDE_PHASE_LR = 5e-7
    
    INPUT_VECTOR = 4
    OUTPUT_VECTOR = 4
    HIDDEN_LAYERS = 8
    HIDDEN_WIDTH = 128
    SPATIAL_FOURIER_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

    COLLOCATION_REFRESH_EVERY = 200
    PDE_DIAGNOSTIC_EVERY = 250
    SUPERVISED_REFRESH_EVERY = 250

    USE_FROZEN_NUT_LAPLACIAN = True

    STRATIFIED_SAMPLING = True

    SEED = 0
    PRINT_EVERY = 1
    PLOT_EVERY = 100
    VALIDATION_EVERY = 20

    CHECKPOINT_SAVE_EVERY = 100
    AUTO_RESUME = True

    COLUMN_MAP = {
        "Points_0": "x",
        "Points_1": "y",
        "Points_2": "z",
        "U_0": "u",
        "U_1": "v",
        "U_2": "w",
        "p": "p",
        "nut": "nut",
        "k": "k"
    }
    
    DYNAMIC_VISCOSITY = 1.816e-5
