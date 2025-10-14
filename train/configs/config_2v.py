 # Logging Info
WORK_DIR = "" # Leave empty
EXP_ID = ""
PROJECT_NAME = "XFACTOR-NVS"


DATA_SEED = 0 

# Dimension of pose latents
POSE_DIM = 256

 
L_WEIGHTS = { "w_l1": {"value": 1.0, "warmup_steps": None, "s": None},
              "w_lpips": {"value": 0.5, "warmup_steps": None, "s": None}
}

# This is the stereo-monocular model, we're working with two images
NUM_SAMPLES = 2
MAX_VIEWS = 2

# Dataloader
NUM_WORKERS = 16
NUM_THREADS = 2
WORKER_BUFFER_SIZE = 4
PREFETCH_FACTOR = 4


FRAME_WARMUP_ITERS = 50000
FRAME_WARMUP_DELAY = 0

# These paths must be set 
RE10K_PATH = "/data/scene-rep/datasets/real_estate_10k/LVSM_ver/"
DL3DV_PATH = "/data/scene-rep/dust3r/loop/datasets/dl3dv/"
MVIMGNET_PATH = "/data/scene-rep/datasets/mv_img_net/mv_img_net_1/data/"
CO3DV2_PATH = "/data/scene-rep/datasets/co3dv2/"

# Could probably be more agressive than these baselines
DATASETS = { "co3dv2": {"min_win_len": None, 
                        "max_win_len_begin": 9,
                        "max_win_len_end": 20,
                        "sample_type": "window",
                        "path": CO3DV2_PATH,
                        "upweight": False,
                        "data_weight": 1},
             "dl3dv":  {"min_win_len": None,
                        "max_win_len_begin": 5,
                        "max_win_len_end": 10,
                        "sample_type": "window",
                        "path": DL3DV_PATH,
                        "upweight": False,
                        "data_weight": 1},
             "re10k":  {"min_win_len": None,
                        "max_win_len_begin": 40,
                        "max_win_len_end": 100,
                        "sample_type": "window",
                        "path": RE10K_PATH,
                        "upweight": False,
                        "data_weight": 1},                        
             "mvimgnet": {"min_win_len": None,
                          "max_win_len_begin": 5,
                          "max_win_len_end": 10,
                          "sample_type": "window",
                          "path": MVIMGNET_PATH,
                          "upweight": False,
                          "data_weight": 1}
           }




AUG = { "pose": True,
        "jitter": {"p_apply": 0.9,
                   "d_bright": 0.2,
                   "d_cont": 0.2,
                   "d_sat": 0.2,
                   "d_hue": 0.05},
       "blur":   {"p_apply": 0.3,
                  "min_sigma": 0.1,
                  "max_sigma": 2.0,
                  "kernel_size": 7}
        }



# Use checkpointing 
CHECKPOINT = True

# Mixed precision training
MIXED = True

# TRUE_BATCH_SIZE is actual batch size
# Set BATCH_SIZE to be some divisor of TRUE_BATCH_SIZE for automatic gradient accumulation
BATCH_SIZE      = 256
TRUE_BATCH_SIZE = 256


INPUT_SIZE = (256, 256)  

# Model Params
FEATURES = 1024
PATCH_SIZE = 16 
NUM_HEADS = 16 

POSE_LAYERS = 8 
RENDER_LAYERS = 8 


# Hyperparams
ADAM_B1 = 0.9
ADAM_B2 = 0.95 

WEIGHT_DECAY = 5.0e-3 
MASK_LAYER_NORM = True

NUM_TRAIN_STEPS = 100000
STOP_AFTER      = 100000
INIT_LR         = 1.0e-12 
LR              = 4.0e-4 
END_LR          = 1.0e-4
WARMUP_STEPS    = 4000 


EVAL_EVERY     = 50000
LOG_LOSS_EVERY = 100
VIZ_EVERY      = 10000
VIZ_SIZE       = (128, 128)
NUM_EVAL_STEPS = 20
NUM_PCA = 12

CHECKPOINT_EVERY = 2000

