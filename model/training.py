from datetime import datetime
import os

import numpy as np
import optuna
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score

from data_preparation import create_eeg_dataset
from model import create_model, create_time_distributed_lstm_cnn
from data_preparation import get_data
from params import EPOCHS, SEGMENT_COLUMNS, SEGMENT_ROWS, KFOLD_N_SPLITS, THRESHOLD, COMMON_CHANNELS, TRAIN_DATA_FILE

input_shape = (SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS))

data_path = TRAIN_DATA_FILE
if not os.path.exists(data_path):
    data = get_data()
    dump(data, data_path)
    print(f"Data was saved to file {data_path}")
else:
    print(f"File {data_path} already exists and will be loaded.")
    with open(data_path, 'rb') as f:
        data = load(f)

labels = np.array([sample['label'] for sample in data])
print(f"Data loaded. Number of samples: {len(data)}")

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # alternatively other metrics e.g. ‘val_loss’, ‘val_f1_score’
    patience=5,
    restore_best_weights=True,
    start_from_epoch=5
)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# Callback to print layer outputs
class PrintLayerOutput(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Access the first batch of validation data
        val_batch = next(iter(self.validation_data))
        val_batch_data, _ = val_batch

        for layer in self.model.layers:
            if 'input' not in layer.name:
                intermediate_layer_model = models.Model(inputs=self.model.input, outputs=layer.output)
                intermediate_output = intermediate_layer_model.predict(val_batch_data)
                print(f"Layer: {layer.name}, Output shape: {intermediate_output.shape}")


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
    batch_size = trial.suggest_categorical('batch_size', [3])

    print(f"Trial Parameters: learning_rate={learning_rate}, batch_size={batch_size}")

    accuracies = []
    skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)
    for fold, (train_index, val_index) in enumerate(skf.split(data, labels)):
        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]

        steps_per_epoch_train = len(train_data) // batch_size
        steps_per_epoch_val = len(val_data) // batch_size
        train_ds = create_eeg_dataset(train_data, batch_size=batch_size)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size)

        # DEBUG
        # for element in val_ds.take(2):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)
        # for element in train_ds.take(2):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)

        # TODO: Turn on strategy and Disable debug mode (args: debug, run_eagerly)
        with strategy.scope():
            model = create_model(trial, input_shape, debug=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy', metrics=[
                    'accuracy',
                    tf.keras.metrics.Recall(),  # Sensitivity/Recall
                    tf.keras.metrics.SpecificityAtSensitivity(sensitivity=THRESHOLD),
                    tf.keras.metrics.F1Score(threshold=THRESHOLD, average='micro')
                ], run_eagerly=False)
        model.summary()

        # DEBUG
        # print_layer_output = PrintLayerOutput(val_ds)
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_ds,
            validation_steps=steps_per_epoch_val,
            verbose=2,
            callbacks=[tensorboard_callback, early_stopping_callback]
            # command to run tensorBoard: tensorboard --logdir=logs/fit
        )

        y_val_pred = model.predict(val_ds, steps=steps_per_epoch_val)
        y_val_pred = (np.array(y_val_pred) >= THRESHOLD).astype(int).flatten()
        y_val_true = np.array([sample["label"] for sample in val_data]).astype(int)
        # kappa = cohen_kappa_score(y_val_true, y_val_pred)
        # TODO: Check if kappa score is correct
        # kappa_v2 = cohen_kappa_score([0,1,1,1], [0,1,1,0])
        # print(f"Cohen's Kappa: {kappa}")

        val_accuracy = np.mean(history.history['val_accuracy'])
        trial.report(val_accuracy, step=fold)

        # Checking whether the trial should be terminated
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        accuracies.append(max(history.history['val_accuracy']))

    return float(np.mean(accuracies))
