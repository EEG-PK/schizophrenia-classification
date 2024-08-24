from typing import Tuple

import optuna
import tensorflow as tf
from tensorflow.keras.regularizers import l2


def create_model(trial, input_shape, debug=False):
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 5)
    filters_list, filter_size_list, strides_conv_list = [], [], []
    pool_size_list, strides_pool_list = [], []

    height, width = input_shape[0], input_shape[1]

    for i in range(n_conv_layers):
        filters = trial.suggest_int(f'filters_{i+1}', 16, 128, step=16)

        max_filter_size = min(height, width) // 2
        if max_filter_size < 2:
            raise optuna.exceptions.TrialPruned(f"Filter size is too large for the current dimensions at layer {i+1}.")
        filter_size = trial.suggest_int(f'filter_size_{i+1}', 2, max_filter_size)
        strides_conv = trial.suggest_int(f'strides_conv_{i+1}', 1, 2)

        height = (height - filter_size) // strides_conv + 1
        width = (width - filter_size) // strides_conv + 1

        if height <= 0 or width <= 0:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after Conv2D at layer {i+1}.")

        pool_size = trial.suggest_int(f'pool_size_{i+1}', 2, 3)
        strides_pool = trial.suggest_int(f'strides_pool_{i+1}', 1, 2)

        height = (height - pool_size) // strides_pool + 1
        width = (width - pool_size) // strides_pool + 1

        if height <= 0 or width <= 0:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after MaxPooling at layer {i+1}.")

        filters_list.append(filters)
        filter_size_list.append(filter_size)
        strides_conv_list.append(strides_conv)
        pool_size_list.append(pool_size)
        strides_pool_list.append(strides_pool)

    lstm_units = trial.suggest_int('lstm_units', 64, 256, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    model = create_time_distributed_lstm_cnn(
        input_shape=input_shape,
        n_conv_layers=n_conv_layers,
        filters_list=filters_list,
        filter_size_list=filter_size_list,
        strides_conv_list=strides_conv_list,
        pool_size_list=pool_size_list,
        strides_pool_list=strides_pool_list,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        debug=debug
    )

    return model


def create_time_distributed_lstm_cnn(input_shape: Tuple[int, int, int],
                                     n_conv_layers, filters_list, filter_size_list,
                                     strides_conv_list, pool_size_list, strides_pool_list, lstm_units,
                                     dropout_rate, l2_reg,
                                     debug=False) -> tf.keras.Model:
    """
    Creates a Time-Distributed CNN-LSTM model for processing EEG signal segments,
    with hyperparameters optimized by Optuna

    :param input_shape: Tuple representing the shape of a single EEG segment (width, height, channels).
    :param debug: Boolean to enable/disable debug print statements.
    :return: A compiled Keras Model ready for training.
    """
    inputs = tf.keras.Input(shape=(None, *input_shape))

    # Hyperparameters from Optuna
    # n_conv_layers = trial.suggest_int('n_conv_layers', 1, 8)
    # filters_list = [trial.suggest_int(f'filters_{i+1}', 16, 128, step=16) for i in range(n_conv_layers)]
    # filter_size_list = [trial.suggest_int(f'filter_size_{i+1}', 2, 5) for i in range(n_conv_layers)]
    # strides_conv_list = [trial.suggest_int(f'strides_conv_{i+1}', 1, 2) for i in range(n_conv_layers)]
    # pool_size_list = [trial.suggest_int(f'pool_size_{i+1}', 2, 3) for i in range(n_conv_layers)]
    # strides_pool_list = [trial.suggest_int(f'strides_pool_{i+1}', 1, 2) for i in range(n_conv_layers)]
    # lstm_units = trial.suggest_int('lstm_units', 64, 256, step=64)
    # dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    # l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    x = inputs

    if debug:
        print(f"Input shape: {x.shape}")

    for i in range(n_conv_layers):
        filters = filters_list[i]
        filter_size = filter_size_list[i]
        strides_conv = strides_conv_list[i]
        pool_size = pool_size_list[i]
        strides_pool = strides_pool_list[i]

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters, (filter_size, filter_size), strides=strides_conv, activation='relu',
                                   kernel_regularizer=l2(l2_reg)))(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((pool_size, pool_size), strides=strides_pool))(x)

        if debug:
            print(f"After Conv2D layer {i + 1}, shape: {x.shape}")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    if debug:
        print(f"After Flatten, shape: {x.shape}")

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    if debug:
        print(f"After LSTM, shape: {x.shape}")

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
