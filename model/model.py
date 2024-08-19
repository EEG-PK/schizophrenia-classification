from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import joblib

from data_preparation import read_eeg_data, load_and_segment_eeg_data, create_eeg_dataset
from params import KFOLD_N_SPLITS, BATCH_SIZE, SEGMENT_WIDTH, SEGMENT_HEIGHT, CHANNEL_NUMBER, EPOCHS


def create_time_distributed_lstm_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Creates a Time-Distributed CNN-LSTM model for processing EEG signal segments.

    :param input_shape: Tuple representing the shape of a single EEG segment (width, height, channels).
    :return: A compiled Keras Model ready for training.
    """
    inputs = tf.keras.Input(shape=(None, *input_shape))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.LSTM(128)(x)  # Zachowanie sekwencyjności segmentów
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Piklowanie danych - taki mockup danych na ten moment
data_dir = 'model/data'
schizophrenia_files = ['33w1.eea', '088w1.eea', '103w.eea']
health_files = ['088w1.eea', '103w.eea']

eeg_data_schizophrenia = [{'id': fine_name, 'eeg': read_eeg_data(f'{data_dir}/{fine_name}')} for fine_name in
                          schizophrenia_files]
eeg_data_health = [{'id': fine_name, 'eeg': read_eeg_data(f'{data_dir}/{fine_name}')} for fine_name in health_files]

f = open("model/eeg_schizophrenia.pk", "wb")
joblib.dump(eeg_data_schizophrenia, f)
f.close()
f = open("model/eeg_health.pk", "wb")
joblib.dump(eeg_data_health, f)
f.close()

# Wczytywanie danych
train_files_schizophrenia = ["model/eeg_schizophrenia.pk"]
train_files_health = ["model/eeg_health.pk"]

# Segmentacja danych
segmented_train_data_schizophrenia = load_and_segment_eeg_data(train_files_schizophrenia, label=1)
segmented_train_data_health = load_and_segment_eeg_data(train_files_health, label=0)

# Połączenie danych i losowe przemieszanie
segmented_train_data = segmented_train_data_schizophrenia + segmented_train_data_health
np.random.shuffle(segmented_train_data)

# Przygotowanie danych do walidacji krzyżowej
labels = np.array([sample['label'] for sample in segmented_train_data])
skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)

# Przechodzenie przez foldy
for train_index, val_index in skf.split(segmented_train_data, labels):
    train_data = [segmented_train_data[i] for i in train_index]
    val_data = [segmented_train_data[i] for i in val_index]

    # Określamy liczbę kroków na epokę (steps_per_epoch)
    steps_per_epoch_train = len(train_data) // BATCH_SIZE
    steps_per_epoch_val = len(val_data) // BATCH_SIZE

    # Tworzenie datasetów
    train_ds = create_eeg_dataset(train_data, batch_size=BATCH_SIZE)
    val_ds = create_eeg_dataset(val_data, batch_size=BATCH_SIZE)

    # Pobieranie kształtu dla modelu
    input_shape = (SEGMENT_WIDTH, SEGMENT_HEIGHT, CHANNEL_NUMBER)  # Dodajemy wymiar kanałów

    # Tworzenie i kompilacja modelu
    model = create_time_distributed_lstm_cnn(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Trening modelu
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=val_ds,
        validation_steps=steps_per_epoch_val
    )
