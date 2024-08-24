import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from optuna_integration import TFKerasPruningCallback
import tensorflow as tf

from data_preparation import create_eeg_dataset, load_and_segment_eeg_data
from model import create_model
from params import EPOCHS, SEGMENT_COLUMNS, SEGMENT_ROWS, CHANNEL_NUMBER, KFOLD_N_SPLITS

# Wczytywanie danych
train_files_schizophrenia = ["model/eeg_schizophrenia.pk"]
train_files_health = ["model/eeg_health.pk"]

# Segmentacja danych
segmented_train_data_schizophrenia = load_and_segment_eeg_data(train_files_schizophrenia, label=1)
segmented_train_data_health = load_and_segment_eeg_data(train_files_health, label=0)
segmented_train_data = segmented_train_data_schizophrenia + segmented_train_data_health

input_shape = (SEGMENT_ROWS, SEGMENT_COLUMNS, CHANNEL_NUMBER)  # Dodajemy wymiar kanałów

np.random.shuffle(segmented_train_data)

# Przygotowanie danych do walidacji krzyżowej
labels = np.array([sample['label'] for sample in segmented_train_data])
skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)


def objective(trial):
    # Hyperparameters from Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1])

    accuracies = []

    # Przechodzenie przez foldy
    for fold, (train_index, val_index) in enumerate(skf.split(segmented_train_data, labels)):
        train_data = [segmented_train_data[i] for i in train_index]
        val_data = [segmented_train_data[i] for i in val_index]

        # Określamy liczbę kroków na epokę (steps_per_epoch)
        steps_per_epoch_train = len(train_data) // batch_size
        steps_per_epoch_val = len(val_data) // batch_size

        # Tworzenie datasetów
        train_ds = create_eeg_dataset(train_data, batch_size=batch_size)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size)

        model = create_model(trial, input_shape, debug=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_ds,
            validation_steps=steps_per_epoch_val,
            # callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')],
            verbose=2
        )

        # Reporting the result after the fold
        val_accuracy = np.mean(history.history['val_accuracy'])
        trial.report(val_accuracy, step=fold)
        # Checking whether the trial should be terminated
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # val_loss, val_accuracy = model.evaluate(val_ds, steps=steps_per_epoch_val, verbose=0)
        accuracies.append(max(history.history['val_accuracy']))

    # Zwracamy średnią dokładność z k-fold
    return np.mean(accuracies)
