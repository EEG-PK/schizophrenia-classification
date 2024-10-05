from datetime import datetime
import os
from typing import Callable, Dict, Any

import numpy as np
import optuna
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score

from data_preparation import create_eeg_dataset
from data_preparation import get_data
from params import EPOCHS, SEGMENT_COLUMNS, SEGMENT_ROWS, KFOLD_N_SPLITS, THRESHOLD, COMMON_CHANNELS, TRAIN_DATA_FILE, \
    MODELS

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
    monitor='val_loss',  # alternatively other metrics e.g. ‘accuracy’, ‘val_f1_score’
    patience=5,
    restore_best_weights=True,
    start_from_epoch=5
)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def check_cnn2d_dim(height: int,
                     width: int,
                     n_conv_layers: int,
                     padding: str,
                     filter_size: int,
                     strides_conv: int,
                     pool_size: int) -> int:
    """Check the dimensions of a 2D CNN after a specified number of convolutional layers.

    This function calculates the height and width of the feature map
    after each convolutional and pooling layer in a 2D CNN. It checks
    if the dimensions remain valid (greater than zero) after processing
    the specified number of layers. If the dimensions become invalid,
    it returns the index of the layer where the dimensions become non-positive.

    :param height: The initial height of the input feature map.
    :param width: The initial width of the input feature map.
    :param n_conv_layers: The number of convolutional layers to process.
    :param padding: The padding method to be used, either 'same' or 'valid'.
    :param filter_size: The size of the convolutional filter.
    :param strides_conv: The stride value for the convolutional layers.
    :param pool_size: The size of the pooling window.

    :return: The index of the layer at which dimensions become non-positive,
             or -1 if all layers maintain valid dimensions.
    """
    for i in range(n_conv_layers):
        if padding == 'same':
            height = height // strides_conv
            width = width // strides_conv
            height = height // pool_size
            width = width // pool_size
        else:
            height = (height - filter_size) // strides_conv + 1
            width = (width - filter_size) // strides_conv + 1
            height = (height - pool_size) // pool_size + 1
            width = (width - pool_size) // pool_size + 1

        print(f"After layer {i + 1}, height: {height}, width: {width}")
        if height <= 0 or width <= 0:
            return i
    return -1


def check_cnn3d_dim(depth: int,
                     height: int,
                     width: int,
                     n_conv_layers: int,
                     padding: str,
                     filter_size: int,
                     strides_conv: int,
                     pool_size: int) -> int:
    """Check the dimensions of a 3D CNN after a specified number of convolutional layers.

    This function calculates the depth, height, and width of the feature map
    after each convolutional and pooling layer in a 3D CNN. It checks
    if the dimensions remain valid (greater than zero) after processing
    the specified number of layers. If the dimensions become invalid,
    it returns the index of the layer where the dimensions become non-positive.

    :param depth: The initial depth of the input feature map.
    :param height: The initial height of the input feature map.
    :param width: The initial width of the input feature map.
    :param n_conv_layers: The number of convolutional layers to process.
    :param padding: The padding method to be used, either 'same' or 'valid'.
    :param filter_size: The size of the convolutional filter.
    :param strides_conv: The stride value for the convolutional layers.
    :param pool_size: The size of the pooling window.

    :return: The index of the layer at which dimensions become non-positive,
             or -1 if all layers maintain valid dimensions.
    """
    for i in range(n_conv_layers):
        if padding == 'same':
            depth = depth // strides_conv
            height = height // strides_conv
            width = width // strides_conv
        else:
            depth = (depth - filter_size) // strides_conv + 1
            height = (height - filter_size) // strides_conv + 1
            width = (width - filter_size) // strides_conv + 1

        depth = depth // 1
        height = height // pool_size
        width = width // pool_size

        print(f"After layer {i + 1}, depth: {depth}, height: {height}, width: {width}")
        if depth <= 0 or height <= 0 or width <= 0:
            return i
    return -1


# DEBUG
# Callback to print layer outputs
# class PrintLayerOutput(callbacks.Callback):
#     def __init__(self, validation_data):
#         super().__init__()
#         self.validation_data = validation_data
#
#     def on_epoch_end(self, epoch, logs=None):
#         # Access the first batch of validation data
#         val_batch = next(iter(self.validation_data))
#         val_batch_data, _ = val_batch
#
#         for layer in self.model.layers:
#             if 'input' not in layer.name:
#                 intermediate_layer_model = models.Model(inputs=self.model.input, outputs=layer.output)
#                 intermediate_output = intermediate_layer_model.predict(val_batch_data)
#                 print(f"Layer: {layer.name}, Output shape: {intermediate_output.shape}")


def k_fold_training(trial: optuna.Trial,
                    model_type: str,
                    batch_size: int,
                    learning_rate: float) -> float:
    """Perform k-fold cross-validation training on a specified model.

    This function utilizes stratified k-fold cross-validation to train
    the model and evaluate its performance across multiple folds.
    It employs various metrics, including accuracy, recall, specificity,
    and F1 score. Early stopping and model pruning are supported
    through the use of Optuna.

    :param trial: An Optuna trial object for hyperparameter optimization.
    :param model_type: The type of model to be created and trained.
                       Must be one of the following: 'cnn_lstm', 'cnn3d'.
    :param batch_size: The number of samples per gradient update.
    :param learning_rate: The learning rate for the optimizer.

    :return: The average validation accuracy across all k-folds.

    :raises optuna.exceptions.TrialPruned: If the trial is pruned based
        on intermediate results.

    :note:
        - The function assumes the existence of global variables:
          `data`, `labels`, `KFOLD_N_SPLITS`, `EPOCHS`, `THRESHOLD`,
          `tensorboard_callback`, `early_stopping_callback`, and `strategy`.
        - TensorFlow/Keras is used for model training and evaluation.
    """
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
            model = create_model(trial, model_type, debug=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy', metrics=[
                    'accuracy',
                    tf.keras.metrics.Recall(),
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
            callbacks=[tensorboard_callback, early_stopping_callback]  # command to run tensorBoard: tensorboard --logdir=logs/fit
        )

        # y_val_pred = model.predict(val_ds, steps=steps_per_epoch_val)
        # y_val_pred = (np.array(y_val_pred) >= THRESHOLD).astype(int).flatten()
        # y_val_true = np.array([sample["label"] for sample in val_data]).astype(int)
        # kappa = cohen_kappa_score(y_val_true, y_val_pred)
        # TODO: Check if kappa score is correct
        # kappa_v2 = cohen_kappa_score([0,1,1,1], [0,1,1,0])
        # print(f"Cohen's Kappa: {kappa}")

        val_accuracy = np.mean(history.history['val_accuracy'])
        trial.report(val_accuracy, step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        accuracies.append(max(history.history['val_accuracy']))

    return float(np.mean(accuracies))


# TODO: Create from it builder/class or something..
def create_model(trial: optuna.Trial, model_type: str, debug: bool = False) -> tf.keras.Model:
    """Create a Keras model based on the specified model type and hyperparameters.

    This function uses Optuna to suggest hyperparameters for the model.
    The function also checks the model dimensions for both 2D and 3D CNNs, ensuring they are valid
    for the given input shape. Depending on the specified model type,
    it initializes the appropriate model from the `MODELS` dictionary.

    :param trial: An Optuna trial object for hyperparameter optimization.
    :param model_type: The type of model to be created. Must be one of:
                       'cnn_lstm' or 'cnn3d'.
    :param debug: Flag indicating whether to enable debug mode (default is False).

    :return: A Keras model instance configured with the suggested hyperparameters.

    :raises optuna.exceptions.TrialPruned: If the dimensions of the model
        become invalid after adding a layer.
    :raises ValueError: If an unknown model_type is specified.

    :note:
        - This function assumes the existence of global variables:
          `input_shape` and `MODELS`.
    """
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 9)
    filters = trial.suggest_int('filters', 8, 80, step=8)
    # filter_size = trial.suggest_int('filter_size', 2, 5)
    # strides_conv = trial.suggest_int('strides_conv', 1, 2)
    filter_size = 3
    strides_conv = 1
    pool_size = 2
    strides_pool = 2
    padding = 'same'
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    model_params: Dict[str, Any] = {
        'input_shape': input_shape,
        'n_conv_layers': n_conv_layers,
        'filters': filters,
        'filter_size': filter_size,
        'strides_conv': strides_conv,
        'pool_size': pool_size,
        'strides_pool': strides_pool,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
        'padding': padding,
        'debug': debug
    }

    if model_type == 'cnn_lstm':
        model_params['lstm_units'] = trial.suggest_int('lstm_units', 6, 192, step=6)

        # Check dimensions for 2D CNN model
        bad_conv_layer_number = check_cnn2d_dim(input_shape[0], input_shape[1], n_conv_layers, padding, filter_size,
                                                strides_conv, pool_size)
        if bad_conv_layer_number != -1:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after layer {bad_conv_layer_number + 1}.")
    elif model_type == 'cnn3d':
        # Check dimensions for 3D CNN model
        bad_conv_layer_number = check_cnn3d_dim(input_shape[2], input_shape[0], input_shape[1], n_conv_layers, padding,
                                                filter_size, strides_conv, pool_size)
        if bad_conv_layer_number != -1:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after layer {bad_conv_layer_number + 1}.")

    try:
        return MODELS[model_type](**model_params)
    except KeyError:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_objective(model_type: str) -> Callable[[optuna.Trial], float]:
    """Create an objective function for Optuna optimization based on the specified model type.

    This function generates an objective function tailored for a specific
    model type, which can be used in the Optuna optimization process.
    The objective function suggests hyperparameters such as learning rate
    and batch size, then evaluates the model's performance using k-fold
    cross-validation.

    :param model_type: The type of model to be optimized. Must be one of:
                       'cnn_lstm' or 'cnn3d'.

    :return: A callable objective function that takes an Optuna trial
             as input and returns the evaluation score (validation accuracy).

    :raises ValueError: If an unknown model_type is specified.

    :note:
        - The function assumes that the `k_fold_training` function is
          available and correctly configured to handle the specified model type.
    """
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [3])
        return k_fold_training(trial, model_type, batch_size, learning_rate)

    return objective
