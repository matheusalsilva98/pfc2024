# Training hyperparameters
NUM_CHANNELS = 4
NUM_CLASSES = 4
VAL_PERCENT = 0.20
TEST_PERCENT = 0.10
BATCH_SIZE = 16
NUM_WORKERS = 12
LEARNING_RATE = 1e-3
PIN_MEMORY = True
DROP_LAST = True
PREFETCH_FACTOR = 2 * BATCH_SIZE

# Dataset
IMGS_DIR = '/mnt/data/borba/pfc_2024/imagens_dataset_pfc/images_LOCAL_CONFIG'
MASKS_DIR = '/mnt/data/borba/pfc_2024/imagens_dataset_pfc/masks_LOCAL_CONFIG'

# Compute related
ACCELERATOR = 'cuda'
DEVICES = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 200
PRECISION = 32  
