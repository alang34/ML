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
            "wake_start_x": 3.83,
        },
        {
            "path": SCRIPT_DIR / "Data/Case2/83.csv",
            "u_ref": 82.6,
            "l_ref": 3.83,
            "cross_ref":48.7,
            "rho_ref": 0.8499,
            "case_id": "speed_82_6",
            "wake_start_x": 3.83,
        },
    ]

    EPOCHS = 500000
    PRETRAIN_EPOCHS = 20000

    LR = 5e-5
    LR_DECAY_HALF_LIFE = 5000
    LR_MIN = 1e-5
    PDE_RAMP_EPOCHS = 15000
    PDE_MAX_GRAD_RATIO = 0.3
    PDE_GRADIENT_EVERY = 50
    PDE_LAMBDA_MAX_GROWTH_FACTOR = 2.0
    BOUNDARY_LOSS_WEIGHT = 1.0
    BOUNDARY_POINTS_PER_GROUP = 256
    BOUNDARY_WARMUP_STEPS = 200

    N_TRAIN_POINTS = 50000
    N_VAL_POINTS = 50000
    SUPERVISED_SAMPLING_FIELD_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
    SUPERVISED_PERSISTENT_FRACTION = 0.05
    SUPERVISED_NEAR_BODY_DISTANCE = 0.10
    SUPERVISED_WAKE_DEFICIT = 0.05
    VORTICITY_LOSS_WEIGHT = 0.05
    VORTICITY_POINTS_PER_CASE = 64
    LOSS_GRADIENT_DIAGNOSTIC_EVERY = 250

    N_COLLOCATION = 8000
    
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
    CHECKPOINT_DIR = "checkpoints/vector_potential_boundaries"
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
