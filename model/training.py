import random
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

from model.data_preparation import create_eeg_dataset, get_data
from model.params import EPOCHS, KFOLD_N_SPLITS, THRESHOLD, \
    TRAIN_DATA_PATH, MODELS, DATASETS_DIR, DATASET_DIR, SCHIZO_DUMP_FILE, HEALTH_DUMP_FILE, SEGMENTS_SPLIT, \
    DATASETS_MELT, STORAGE_NAME, DATA_SAMPLE_SHAPE

train_files_health = f"{DATASETS_DIR}/{DATASET_DIR}/{HEALTH_DUMP_FILE}"
train_files_schizophrenia = f"{DATASETS_DIR}/{DATASET_DIR}/{SCHIZO_DUMP_FILE}"

data_path = TRAIN_DATA_PATH
if not os.path.exists(data_path):
    data = get_data(train_files_health, train_files_schizophrenia)
    dump(data, data_path)
    print(f"Data was saved to file {data_path}")
else:
    print(f"File {data_path} already exists and will be loaded.")
    with open(data_path, 'rb') as f:
        data = load(f)
print(f"Data loaded. Number of samples: {len(data)}")

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


def flat_samples(data_samples, shuffle=True):
    all_segments = []
    for sample in data_samples:
        if SEGMENTS_SPLIT:
            for segment in sample["segments"]:
                all_segments.append((segment, sample["label"]))
        else:
            all_segments.append((sample["segments"], sample["label"]))
    if shuffle:
        random.shuffle(all_segments)
    return all_segments


def k_fold_training(trial: optuna.Trial,
                    model_type: str) -> float:
    """Perform k-fold cross-validation training on a specified model.

    This function utilizes stratified k-fold cross-validation to train
    the model and evaluate its performance across multiple folds.
    It employs various metrics, including accuracy, recall, specificity,
    and F1 score. Early stopping and model pruning are supported
    through the use of Optuna.

    :param trial: An Optuna trial object for hyperparameter optimization.
    :param model_type: The type of model to be created and trained.

    :return: The average validation accuracy across all k-folds.

    :raises optuna.exceptions.TrialPruned: If the trial is pruned based
        on intermediate results.

    :note:
        - The function assumes the existence of global variables:
          `data`, `labels`, `KFOLD_N_SPLITS`, `EPOCHS`, `THRESHOLD`,
          `tensorboard_callback`, `early_stopping_callback`, and `strategy`.
        - TensorFlow/Keras is used for model training and evaluation.
    """
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    learning_rate = 4.267e-05
    # batch_size = trial.suggest_categorical('batch_size', [12, 36, 60])
    batch_size = 36

    # model_params = get_hyperparams(trial, model_type, debug=True)
    # hyperparams_str = '__'.join([f"{key}_{value}" for key, value in trial.params.items()])

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        start_from_epoch=5
    )

    if DATASETS_MELT:
        all_data = flat_samples(data)
        labels = np.array([label for sample, label in all_data])
    else:
        all_data = data
        labels = np.array([sample['label'] for sample in all_data])

    accuracies = []
    skf = StratifiedKFold(n_splits=KFOLD_N_SPLITS)
    for fold, (train_index, val_index) in enumerate(skf.split(all_data, labels)):
        print(f"Fold {fold + 1}")
        train_data = [all_data[i] for i in train_index]
        val_data = [all_data[i] for i in val_index]
        if not DATASETS_MELT:
            train_data = flat_samples(train_data)
            val_data = flat_samples(val_data)

        train_ds = create_eeg_dataset(train_data, batch_size=batch_size, shuffle=True, generator=True, repeat=False)
        val_ds = create_eeg_dataset(val_data, batch_size=batch_size, generator=True, repeat=False)

        # DEBUG
        # for element in val_ds.take(10):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)
        # for element in train_ds.take(10):
        #     frames, label = element
        #     print("Frames shape:", frames.shape)
        #     print("Label:", label.numpy())
        #     print("Label shape:", label.shape)

        # log_dir = os.path.join(
        #     "logs", "fit", STORAGE_NAME, model_type,
        #     hyperparams_str, session_timestamp, f"fold_{fold + 1}"
        # )
        log_dir = os.path.join(
            "logs", "fit", STORAGE_NAME, model_type, session_timestamp, f"fold_{fold + 1}"
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        with strategy.scope():
            metrics = [
                'accuracy',
                tf.keras.metrics.Recall(),
                tf.keras.metrics.SpecificityAtSensitivity(sensitivity=THRESHOLD),
                tf.keras.metrics.F1Score(threshold=THRESHOLD, average='micro')
            ]

            try:
                model = MODELS[model_type](trial=trial, input_shape=DATA_SAMPLE_SHAPE, learning_rate=learning_rate,
                                           metrics=metrics, debug=False)
            except KeyError as e:
                print(f"Unexpected error: {trial.number}: {str(e)}")
                raise ValueError(f"Unknown model_type: {model_type}")
            except ValueError as e:
                print(f"Error during creating the model: {trial.number}: {str(e)}")
                raise optuna.exceptions.TrialPruned(f"Probably invalid dimensions.")

        model.summary()

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            verbose=2,
            callbacks=[tensorboard_callback, early_stopping_callback]
        )

        val_accuracy = np.mean(history.history['val_accuracy'])
        trial.report(val_accuracy, step=fold)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        accuracies.append(max(history.history['val_accuracy']))

    return float(np.mean(accuracies))


# TODO: Create from it builder/class or something..
def get_hyperparams(trial: optuna.Trial, model_type: str, debug: bool = False, input_shape=DATA_SAMPLE_SHAPE) -> Dict[str, Any]:
    """Create a Keras model based on the specified model type and hyperparameters.

    This function uses Optuna to suggest hyperparameters for the model.
    The function also checks the model dimensions for both 2D and 3D CNNs, ensuring they are valid
    for the given input shape. Depending on the specified model type,
    it initializes the appropriate model from the `MODELS` dictionary.

    :param trial: An Optuna trial object for hyperparameter optimization.
    :param model_type: The type of model to be created.
    :param debug: Flag indicating whether to enable debug mode (default is False).

    :return: A Keras model instance configured with the suggested hyperparameters.

    :raises optuna.exceptions.TrialPruned: If the dimensions of the model
        become invalid after adding a layer.
    :raises ValueError: If an unknown model_type is specified.

    :note:
        - This function assumes the existence of global variables:
          `input_shape` and `MODELS`.
    """

    model_params: Dict[str, Any] = {
        'input_shape': input_shape,
        'debug': debug
    }

    if model_type == 'cnn_lstm':
        filter_size = 3
        strides_conv = 1
        pool_size = 2
        strides_pool = 2
        padding = 'same'
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 9)
        model_params: Dict[str, Any] = {
            'input_shape': input_shape,
            'n_conv_layers': n_conv_layers,
            'filters': trial.suggest_int('filters', 8, 80, step=8),
            'filter_size': filter_size,
            'strides_conv': strides_conv,
            'pool_size': pool_size,
            'strides_pool': strides_pool,
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            'padding': padding,
            'debug': debug,
            'lstm_units': trial.suggest_int('lstm_units', 6, 192, step=6)
        }

        # Check dimensions for 2D CNN model
        bad_conv_layer_number = check_cnn2d_dim(input_shape[0], input_shape[1], n_conv_layers, padding, filter_size,
                                                strides_conv, pool_size)
        if bad_conv_layer_number != -1:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after layer {bad_conv_layer_number + 1}.")

    elif model_type == 'cnn3d':
        filter_size = 3
        strides_conv = 1
        pool_size = 2
        strides_pool = 2
        padding = 'same'
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 9)
        model_params: Dict[str, Any] = {
            'input_shape': input_shape,
            'n_conv_layers': n_conv_layers,
            'filters': trial.suggest_int('filters', 8, 80, step=8),
            'filter_size': filter_size,
            'strides_conv': strides_conv,
            'pool_size': pool_size,
            'strides_pool': strides_pool,
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            'padding': padding,
            'debug': debug
        }
        # Check dimensions for 3D CNN model
        bad_conv_layer_number = check_cnn3d_dim(input_shape[2], input_shape[0], input_shape[1], n_conv_layers, padding,
                                                filter_size, strides_conv, pool_size)
        if bad_conv_layer_number != -1:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after layer {bad_conv_layer_number + 1}.")

    elif model_type == 'cnn_lstm_prepared':
        model_params['lstm_units'] = trial.suggest_int('lstm_units', 6, 192, step=6)
        model_params['merge_layer'] = trial.suggest_categorical('merge_layer', ['avg_pool', 'max_pool', 'flatten'])
        model_params['merge_layer_lstm'] = trial.suggest_categorical('merge_layer', ['pool2d', 'flatten'])

    elif model_type == 'cnn_prepared':
        model_params['merge_layer'] = trial.suggest_categorical('merge_layer', ['avg_pool', 'flatten'])
        model_params['padding'] = trial.suggest_categorical('padding', ['valid', 'same'])
        model_params['filters'] = []
        for i in range(5):
            model_params['filters'].append(trial.suggest_int('filter_number_{}'.format(i), 32, 416, step=64))
        model_params['kernel_sizes'] = []
        for i in range(5):
            model_params['kernel_sizes'].append(trial.suggest_int('kernel_sizes_{}'.format(i), 3, 9, step=1))
        model_params['pool_sizes'] = []
        for i in range(3):
            model_params['pool_sizes'].append(trial.suggest_int('pool_sizes_{}'.format(i), 2, 4, step=1))
        model_params['dense_units'] = []
        for i in range(3):
            model_params['dense_units'].append(trial.suggest_int('dense_units_{}'.format(i), 192, 4096, step=50))

    elif model_type == 'efficientnet':
        model_params['dropout_rate'] = trial.suggest_categorical('dropout_rate', [0.1, 0.3, 0.5])

    return model_params


def create_objective(model_type: str) -> Callable[[optuna.Trial], float]:
    """Create an objective function for Optuna optimization based on the specified model type.

    This function generates an objective function tailored for a specific
    model type, which can be used in the Optuna optimization process.
    The objective function suggests hyperparameters such as learning rate
    and batch size, then evaluates the model's performance using k-fold
    cross-validation.

    :param model_type: The type of model to be optimized.

    :return: A callable objective function that takes an Optuna trial
             as input and returns the evaluation score (validation accuracy).

    :raises ValueError: If an unknown model_type is specified.

    :note:
        - The function assumes that the `k_fold_training` function is
          available and correctly configured to handle the specified model type.
    """

    def objective(trial):
        # learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        # learning_rate = trial.suggest_categorical('learning_rate', [3.6131821360463974e-05])
        # batch_size = trial.suggest_categorical('batch_size', [10, 20, 30, 40])
        # model_type = trial.suggest_categorical('model_type', ['cnn_lstm', 'cnn3d', 'cnn_lstm_prepared', 'cnn_prepared'])
        # model =  MODELS[model_type](trial=trial, input_shape=DATA_SAMPLE_SHAPE, learning_rate=learning_rate, debug=True)
        return k_fold_training(trial, model_type)

    return objective


def unfreeze_model(model, learning_rate):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
