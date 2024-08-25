from datetime import datetime

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import cohen_kappa_score

from data_preparation import create_eeg_dataset, load_and_segment_eeg_data
from model import create_model
from params import EPOCHS, SEGMENT_COLUMNS, SEGMENT_ROWS, CHANNEL_NUMBER, KFOLD_N_SPLITS

# Data loading
train_files_schizophrenia = ["model/eeg_schizophrenia.pk"]
train_files_health = ["model/eeg_health.pk"]

# Data segmentation
segmented_train_data_schizophrenia = load_and_segment_eeg_data(train_files_schizophrenia, label=1)
segmented_train_data_health = load_and_segment_eeg_data(train_files_health, label=0)
segmented_train_data = segmented_train_data_schizophrenia + segmented_train_data_health

input_shape = (SEGMENT_ROWS, SEGMENT_COLUMNS, CHANNEL_NUMBER)

np.random.shuffle(segmented_train_data)

# Preparing data for cross-validation
labels = np.array([sample['label'] for sample in segmented_train_data])
skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        specificity = true_negatives / (possible_negatives + K.epsilon())
        return specificity

    def f1_score(y_true, y_pred):
        precision = tf.keras.metrics.Precision()(y_true, y_pred)
        recall = tf.keras.metrics.Recall()(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    # Passing through the folds
    for fold, (train_index, val_index) in enumerate(skf.split(segmented_train_data, labels)):
        train_data = [segmented_train_data[i] for i in train_index]
        val_data = [segmented_train_data[i] for i in val_index]

        steps_per_epoch_train = len(train_data) // batch_size
        steps_per_epoch_val = len(val_data) // batch_size

        # Dataset creation
        train_ds = create_eeg_dataset(train_data, batch_size=batch_size)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size)

        model = create_model(trial, input_shape, debug=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=[
                'accuracy',
                tf.keras.metrics.Recall(),  # Sensitivity/Recall
                specificity,  # Specificity
                f1_score
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
