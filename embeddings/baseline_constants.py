# Batch sizes
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 512

# Data config
IN_MEMORY_DATASET = True
INCLUDE_ANSWER = True
USE_CLASS_WEIGHTS_SAMPLER = True
USE_CLASS_WEIGHTS_LOSS = False

# W&B dashboard logging
USE_WANDB = True
WANDB_PROJECT_NAME = "heterogeneous-GAT-model"
WANDB_RUN_NAME = "baseline_with_answer"  # None for timestamp

# OS
OS_NAME = "linux"  # "windows" or "linux"
NUM_WORKERS = 14

# Training parameters
TRAIN_DATA_PATH = "data/train-6001-qs.pt"
TEST_DATA_PATH = "data/test-6001-qs.pt"
EPOCHS = 30
START_LR = 0.001
GAMMA = 0.85
WARM_START_FILE = "../models/baseline-model-1.pt"

# (Optional) k-fold cross validation
CROSS_VALIDATE = True
FOLD_FILES = ['fold-1-6001-qs.pt', 'fold-2-6001-qs.pt', 'fold-3-6001-qs.pt', 'fold-4-6001-qs.pt', 'fold-5-6001-qs.pt']
PICKLE_PATH_KF = 'baseline_q_a_kf_results.pkl'

# Model architecture 
HIDDEN_CHANNELS = 64
FINAL_MODEL_OUT_PATH = "baseline_with_answer.pt"
SAVE_CHECKPOINTS = True
DROPOUT=0.0