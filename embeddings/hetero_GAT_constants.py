# Batch sizes
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

# Data config
IN_MEMORY_DATASET = False
INCLUDE_ANSWER = False
USE_CLASS_WEIGHTS_SAMPLER = True
USE_CLASS_WEIGHTS_LOSS = False

# W&B dashboard logging
USE_WANDB = False
WANDB_PROJECT_NAME = "heterogeneous-GAT-model"
WANDB_RUN_NAME = "EXP1-run"  # None for timestamp

# OS
OS_NAME = "windows"  # "windows" or "linux"
NUM_WORKERS = 0

# Training parameters
ROOT = "../datav2"
TRAIN_DATA_PATH = "train-4175-qs.pt"
TEST_DATA_PATH = "test-1790-qs.pt"
EPOCHS = 4
START_LR = 0.001
GAMMA = 0.95
WARM_START_FILE = None#"../models/gat_qa_20e_64h_3l.pt"

# (Optional) k-fold cross validation
CROSS_VALIDATE = False
FOLD_FILES = ['fold-1-6001-qs.pt', 'fold-2-6001-qs.pt', 'fold-3-6001-qs.pt', 'fold-4-6001-qs.pt', 'fold-5-6001-qs.pt']
PICKLE_PATH_KF = 'q_kf_results.pkl'

# Model architecture 
MODEL = "GAT"
NUM_LAYERS = 3
HIDDEN_CHANNELS = 64
FINAL_MODEL_OUT_PATH = "gat_fault.pt"
SAVE_CHECKPOINTS = False
DROPOUT=0.0
