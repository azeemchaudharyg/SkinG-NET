# General settings

SEED = 42

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY=1e-4
EPOCHS = 250

# Model settings
K = 10           # Number of propagation steps for APPNP
ALPHA = 0.1      # Teleport probability for APPNP

# Dataset Paths
DATA_PATH = "data/isic_up_resnet152_features_labels.pkl"  # data/isic_up_resnet152_features_labels.pkl - data/isic_tbp_features_labels.pkl

WANDB_PROJECT = "First_GNN"
WANDB_RUN_NAME = "APPNP_Results"
