SEGMENT_SIZE_SEC = 3
SAMPLING_RATE = 128
SEGMENT_COLUMNS = SEGMENT_SIZE_SEC * SAMPLING_RATE
SEGMENT_ROWS = SEGMENT_COLUMNS // 2
EPOCHS = 6
KFOLD_N_SPLITS = 2
THRESHOLD = 0.5
DATASETS_DIR = '../data'
DATASET = 'dataset1'
EXTENSION = 'eea'
# TODO: Change filenames
SCHIZO_DUMP_FILE = 'eeg_schizophrenia.pk'
HEALTH_DUMP_FILE = 'eeg_health.pk'
CHANNEL_NUMBER = 16
COMMON_CHANNELS = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz']
