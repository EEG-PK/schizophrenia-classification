from model.models import model_cnn_lstm, model_cnn3d, model_cnn_lstm_prepared, model_cnn_prepared

SEGMENT_SIZE_SEC = 3
SAMPLING_RATE = 128
IMAGE_SIZE = (224, 224)
COMMON_CHANNELS = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz']
if IMAGE_SIZE:
    SEGMENT_COLUMNS = IMAGE_SIZE[0]
    SEGMENT_ROWS = IMAGE_SIZE[1]
else:
    SEGMENT_COLUMNS = SEGMENT_SIZE_SEC * SAMPLING_RATE
    SEGMENT_ROWS = SEGMENT_COLUMNS // 2
SEGMENT = True  # # If True: Samples are split to segments (every SEGMENT_SIZE_SEC)
SEGMENTS_SPLIT = True  # If True: Segments in one sample are treated as separated classifications in the model
if SEGMENT and not SEGMENTS_SPLIT:
    DATA_SAMPLE_SHAPE = (None, SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS))
else:
    DATA_SAMPLE_SHAPE = (SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS))

EPOCHS = 70
KFOLD_N_SPLITS = 4
THRESHOLD = 0.5

DATASETS_DIR = 'Data'
DATASET_DIR = 'EeaData'
SCHIZO_DUMP_FILE = 'eeg_Eea_ill.pk'
HEALTH_DUMP_FILE = 'eeg_Eea_health.pk'

TRAIN_DATA_PATH = f'{DATASETS_DIR}/data'
STORAGE_NAME = f'model'
if SEGMENT:
    TRAIN_DATA_PATH += f'_segmentation'
    STORAGE_NAME += f'_segmentation'
    if SEGMENTS_SPLIT:
        STORAGE_NAME += f'_split'
if IMAGE_SIZE:
    TRAIN_DATA_PATH += f'_images_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'
    STORAGE_NAME += f'_images_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'
TRAIN_DATA_PATH += f'.pk'
STORAGE_NAME += f'.db'

# MODELS = {"cnn_lstm" : model_cnn_lstm,
#           "cnn3d" : model_cnn3d}
# MODELS = {"cnn_lstm_prepared" : model_cnn_lstm_prepared}
MODELS = {"cnn_prepared" : model_cnn_prepared}
