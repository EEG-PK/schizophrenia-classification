from model.models import model_cnn_lstm, model_cnn3d

SEGMENT_SIZE_SEC = 3
SAMPLING_RATE = 128
SEGMENT_COLUMNS = SEGMENT_SIZE_SEC * SAMPLING_RATE
SEGMENT_ROWS = SEGMENT_COLUMNS // 2
EPOCHS = 70
KFOLD_N_SPLITS = 4
THRESHOLD = 0.5

DATASETS_DIR = '../Preprocessing'
DATASET_DIR = 'EeaData'
SCHIZO_DUMP_FILE = 'eeg_Eea_ill.pk'
HEALTH_DUMP_FILE = 'eeg_Eea_health.pk'
TRAIN_DATA_FILE = 'data.pk'
COMMON_CHANNELS = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz']
MODELS = {"cnn_lstm" : model_cnn_lstm,
          "cnn3d" : model_cnn3d}
