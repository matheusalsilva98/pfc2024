# Training hyperparameters
NUM_CHANNELS = 6
NUM_CLASSES = 4
VAL_PERCENT = 0.20
TEST_PERCENT = 0.10
BATCH_SIZE = 16
PATCH_SIZE = 512
NUM_WORKERS = 12
LEARNING_RATE = 1e-3
PIN_MEMORY = True
DROP_LAST = True
PREFETCH_FACTOR = 2 * BATCH_SIZE

# Training Dataset
TRAIN_IMGS_DIR = "/mnt/data/borba/pfc_2024/imagens_dataset_pfc/dataset6bandas_treino_valid/imgs_treinamento/images6bands_LOCAL_CONFIG"
TRAIN_MASKS_DIR = "/mnt/data/borba/pfc_2024/imagens_dataset_pfc/dataset6bandas_treino_valid/imgs_treinamento/masks6bands_LOCAL_CONFIG"

# Validation Dataset
VALID_IMGS_DIR = "/mnt/data/borba/pfc_2024/imagens_dataset_pfc/dataset6bandas_treino_valid/imgs_validacao/images6bands_LOCAL_CONFIG"
VALID_MASKS_DIR = "/mnt/data/borba/pfc_2024/imagens_dataset_pfc/dataset6bandas_treino_valid/imgs_validacao/masks6bands_LOCAL_CONFIG"

LOG_OUTPUT_PATH = "/mnt/data/borba/pfc_2024/"

# Compute related
ACCELERATOR = 'cuda'
DEVICES = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 100
PRECISION = 32

USE_AUGMENTATIONS = True
