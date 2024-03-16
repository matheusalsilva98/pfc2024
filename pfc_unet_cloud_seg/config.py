# Training hyperparameters
NUM_CHANNELS = 4
NUM_CLASSES = 4
VAL_PERCENT = 0.20
TEST_PERCENT = 0.10
BATCH_SIZE = 4
NUM_WORKERS = 2 
LEARNING_RATE = 1e-3
PIN_MEMORY = True
DROP_LAST = True
PREFETCH_FACTOR = 2 * BATCH_SIZE

# Dataset
IMGS_DIR = 'C:/Users/SE10/Desktop/PFC_reginaldo_dados/train_validate_dataset/images_LOCAL_CONFIG'
MASKS_DIR = 'C:/Users/SE10/Desktop/PFC_reginaldo_dados/train_validate_dataset/masks_LOCAL_CONFIG'

# Compute related
ACCELERATOR = 'gpu'
DEVICES = 1
MIN_EPOCHS = 1
MAX_EPOCHS = 10
PRECISION = 16  
