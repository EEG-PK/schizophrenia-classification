from model.models import model_cnn_lstm, model_cnn3d, model_cnn_lstm_prepared, model_cnn_prepared, efficientnet, \
    model_cnn_search

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
DATASETS_MELT = True
AVERAGE_CHANNELS_3 = False  # If average channels values to create only 3 channels (for RGB models)
CHANNEL_GROUPS = [
    ['F8', 'F4', 'C4', 'Cz'],  # Group 1
    ['F7', 'F3', 'C3', 'P3'],  # Group 2
    ['O2', 'O1', 'P4', 'Pz']   # Group 3
]
FINE_TUNING = False

if SEGMENT and not SEGMENTS_SPLIT:
    if AVERAGE_CHANNELS_3:
        DATA_SAMPLE_SHAPE = (None, SEGMENT_ROWS, SEGMENT_COLUMNS, 3)
    else:
        DATA_SAMPLE_SHAPE = (None, SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS))
else:
    if AVERAGE_CHANNELS_3:
        DATA_SAMPLE_SHAPE = (SEGMENT_ROWS, SEGMENT_COLUMNS, 3)
    else:
        DATA_SAMPLE_SHAPE = (SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS))

NCHW_INPUT_FORMAT = False
if NCHW_INPUT_FORMAT:
    if len(DATA_SAMPLE_SHAPE) == 3:
        DATA_SAMPLE_SHAPE = (DATA_SAMPLE_SHAPE[2], DATA_SAMPLE_SHAPE[0], DATA_SAMPLE_SHAPE[1])
    elif len(DATA_SAMPLE_SHAPE) == 4:
        DATA_SAMPLE_SHAPE = (DATA_SAMPLE_SHAPE[0], DATA_SAMPLE_SHAPE[3], DATA_SAMPLE_SHAPE[1], DATA_SAMPLE_SHAPE[2])

EPOCHS = 100
KFOLD_N_SPLITS = 4
THRESHOLD = 0.5

DATASETS_DIR = 'Data'
DATASET_DIR = 'CsvData'
SCHIZO_DUMP_FILE = 'eeg_Csv_ill.pk'
HEALTH_DUMP_FILE = 'eeg_Csv_health.pk'

TRAIN_DATA_PATH = f'{DATASETS_DIR}/{DATASET_DIR}'
STORAGE_NAME = f'model'
if SEGMENT:
    TRAIN_DATA_PATH += f'_segmentation'
    STORAGE_NAME += f'_segmentation'
    if SEGMENTS_SPLIT:
        STORAGE_NAME += f'_split'
        if DATASETS_MELT:
            STORAGE_NAME += f'_melt'
if IMAGE_SIZE:
    TRAIN_DATA_PATH += f'_images_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'
    STORAGE_NAME += f'_images_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'
if AVERAGE_CHANNELS_3:
    TRAIN_DATA_PATH += f'_3channels'
    STORAGE_NAME += f'_3channels'
TRAIN_DATA_PATH += f'.pk'
STORAGE_NAME += f'.db'

# MODELS = {"cnn_lstm" : model_cnn_lstm,
#           "cnn3d" : model_cnn3d}
MODELS = {"cnn_prepared" : model_cnn_prepared}
# MODELS = {"efficientnet" : efficientnet}
# MODELS = {"model_cnn_search" : model_cnn_search}
