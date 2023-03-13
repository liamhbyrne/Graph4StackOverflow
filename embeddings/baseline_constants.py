# Batch sizes
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

# Data config
IN_MEMORY_DATASET = True
INCLUDE_ANSWER = True
USE_CLASS_WEIGHTS_SAMPLER = True
USE_CLASS_WEIGHTS_LOSS = False

# W&B dashboard logging
USE_WANDB = True
WANDB_PROJECT_NAME = "heterogeneous-GAT-model"
WANDB_RUN_NAME = None  # None for timestamp

# OS
OS_NAME = "windows"  # "windows" or "linux"
NUM_WORKERS = 0

# Training parameters
TRAIN_DATA_PATH = "data/train-6001-qs.pt"
TEST_DATA_PATH = "data/test-6001-qs.pt"
EPOCHS = 30

# (Optional) k-fold cross validation
CROSS_VALIDATE = False
FOLD_FILES = ['fold-1-6001-qs.pt', 'fold-2-6001-qs.pt', 'fold-3-6001-qs.pt', 'fold-4-6001-qs.pt', 'fold-5-6001-qs.pt']

# Model architecture
HIDDEN_CHANNELS = 64
FINAL_MODEL_OUT_PATH = "model.pt"
SAVE_CHECKPOINTS = False
DROPOUT = 0.5
