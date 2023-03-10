# Batch sizes
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

# Data config
IN_MEMORY_DATASET = True
INCLUDE_ANSWER = True

# W&B dashboard logging
USE_WANDB = True
WANDB_PROJECT_NAME = "heterogeneous-GAT-model"
WANDB_RUN_NAME = None  # None for timestamp

NUM_WORKERS = 14

# Training parameters
EPOCHS = 2

# Model architecture 
NUM_LAYERS = 3
HIDDEN_CHANNELS = 64
FINAL_MODEL_OUT_PATH = "model.pt"
SAVE_CHECKPOINTS = False