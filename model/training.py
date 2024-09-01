from datetime import datetime

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score

from data_preparation import create_eeg_dataset, load_and_segment_eeg_data
from model import create_model, create_time_distributed_lstm_cnn
from params import EPOCHS, SEGMENT_COLUMNS, SEGMENT_ROWS, CHANNEL_NUMBER, KFOLD_N_SPLITS, DATASET, DATASETS_DIR, \
    SCHIZO_DUMP_FILE, HEALTH_DUMP_FILE

# Data loading
train_files_schizophrenia = [f"{DATASETS_DIR}/{DATASET}/{SCHIZO_DUMP_FILE}"]
train_files_health = [f"{DATASETS_DIR}/{DATASET}/{HEALTH_DUMP_FILE}"]

# Data segmentation
segmented_train_data_schizophrenia = load_and_segment_eeg_data(train_files_schizophrenia, label=1)
segmented_train_data_health = load_and_segment_eeg_data(train_files_health, label=0)
segmented_train_data = segmented_train_data_schizophrenia + segmented_train_data_health

input_shape = (SEGMENT_ROWS, SEGMENT_COLUMNS, CHANNEL_NUMBER)

np.random.shuffle(segmented_train_data)

# Preparing data for cross-validation
labels = np.array([sample['label'] for sample in segmented_train_data])
skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize hyperparameters for a time-distributed LSTM-CNN model.

    It performs k-fold cross-validation, training the model on each fold and reporting the validation accuracy.

    :param trial: An Optuna trial object that suggests hyperparameters.
    :return: The average validation accuracy across all folds.

    :raises optuna.exceptions.TrialPruned: If the trial should be pruned based on intermediate results.
    """
    # Hyperparameters from Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1])

    accuracies = []

    # Passing through the folds
    for fold, (train_index, val_index) in enumerate(skf.split(segmented_train_data, labels)):
        train_data = [segmented_train_data[i] for i in train_index]
        val_data = [segmented_train_data[i] for i in val_index]

        steps_per_epoch_train = len(train_data) // batch_size
        steps_per_epoch_val = len(val_data) // batch_size

        # Dataset creation
        train_ds = create_eeg_dataset(train_data, batch_size=batch_size)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size)

        # Debug
        # for element in val_ds.take(2):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)

        model = create_model(trial, input_shape, debug=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=[
                'accuracy',
                tf.keras.metrics.Recall(),  # Sensitivity/Recall
                tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.5),
                tf.keras.metrics.F1Score(threshold=0.5, average='micro')
            ])

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_ds,
            validation_steps=steps_per_epoch_val,
            verbose=2,
            callbacks=[tensorboard_callback]  # command to run tensorBoard: tensorboard --logdir=logs/fit
        )

        y_val_pred = model.predict(val_ds)
        y_val_pred = np.round(y_val_pred).astype(int)
        y_val_true = np.concatenate([y for x, y in val_ds], axis=0)

        kappa = cohen_kappa_score(y_val_true, y_val_pred)
        print(f"Cohen's Kappa: {kappa}")

        # Reporting the result after the fold
        val_accuracy = np.mean(history.history['val_accuracy'])
        trial.report(val_accuracy, step=fold)
        # Checking whether the trial should be terminated
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        accuracies.append(max(history.history['val_accuracy']))

    return np.mean(accuracies)


def test_model():
    learning_rate = 0.001
    batch_size = 1
    accuracies = []

    # Passing through the folds
    for fold, (train_index, val_index) in enumerate(skf.split(segmented_train_data, labels)):
        train_data = [segmented_train_data[i] for i in train_index]
        val_data = [segmented_train_data[i] for i in val_index]

        steps_per_epoch_train = len(train_data) // batch_size
        steps_per_epoch_val = len(val_data) // batch_size

        # Dataset creation
        train_ds = create_eeg_dataset(train_data, batch_size=batch_size)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size)

        # Debug
        # for element in val_ds.take(2):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)

        model = create_time_distributed_lstm_cnn(
            input_shape=input_shape,
            n_conv_layers=3,
            filters=64,
            filter_size=3,
            strides_conv=1,
            pool_size=2,
            strides_pool=1,
            lstm_units=5,
            dropout_rate=0.5,
            l2_reg=0.02,
            debug=True
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=[
                'accuracy',
                tf.keras.metrics.Recall(),  # Sensitivity/Recall
                tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.5),
                tf.keras.metrics.F1Score(threshold=0.5, average='micro')
            ])

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_ds,
            validation_steps=steps_per_epoch_val,
            verbose=2,
            callbacks=[tensorboard_callback]  # command to run tensorBoard: tensorboard --logdir=logs/fit
        )

        y_val_pred = model.predict(val_ds)
        y_val_pred = np.round(y_val_pred).astype(int)
        y_val_true = np.concatenate([y for x, y in val_ds], axis=0)

        kappa = cohen_kappa_score(y_val_true, y_val_pred)
        print(f"Cohen's Kappa: {kappa}")

        accuracies.append(max(history.history['val_accuracy']))

    return np.mean(accuracies)
