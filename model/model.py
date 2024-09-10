from typing import List, Tuple

import optuna
import tensorflow as tf


def create_model(
        trial: optuna.Trial,
        input_shape: Tuple[int, int, int],
        debug: bool = False
) -> tf.keras.Model:
    """
    Creates a time-distributed LSTM-CNN model based on hyperparameters suggested by Optuna.

    This function uses the `trial` object from Optuna to suggest hyperparameters for the convolutional layers,
    pooling layers, and the LSTM layer. It also ensures that the dimensions after each convolution and pooling
    operation are valid. If dimensions become invalid, the trial is pruned.

    :param trial: The Optuna trial object used to suggest hyperparameters.
    :param input_shape: The shape of the input data, specified as (height, width, channels).
    :param debug: A flag to indicate if debug mode is enabled. Defaults to False.
    :return: A compiled Keras model with the suggested hyperparameters.
    :raises optuna.exceptions.TrialPruned: If the dimensions after convolution or pooling become invalid.
    """
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    filters = trial.suggest_int('filters', 16, 80, step=16)
    # filter_size = trial.suggest_int('filter_size', 2, 5)
    filter_size = 3
    # strides_conv = trial.suggest_int('strides_conv', 1, 2)
    strides_conv = 1
    # pool_size = trial.suggest_int('pool_size', 2, 3)
    pool_size = 2
    # strides_pool = trial.suggest_int('strides_pool', 1, 2)
    strides_pool = 2
    lstm_units = trial.suggest_int('lstm_units', 64, 192, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    print(f"Trial Parameters: n_conv_layers={n_conv_layers}, filters={filters}, filter_size={filter_size}, "
          f"strides_conv={strides_conv}, pool_size={pool_size}, strides_pool={strides_pool}, "
          f"lstm_units={lstm_units}, dropout_rate={dropout_rate}, l2_reg={l2_reg}")

    height, width = input_shape[0], input_shape[1]
    padding = 'same'  # Using same padding for consistent dimensions

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

        if height <= 0 or width <= 0:
            raise optuna.exceptions.TrialPruned(f"Invalid dimensions after layer {i + 1}.")

        if debug:
            print(f"After layer {i + 1}, height: {height}, width: {width}")

    model = create_time_distributed_lstm_cnn(
        input_shape=input_shape,
        n_conv_layers=n_conv_layers,
        filters=filters,
        filter_size=filter_size,
        strides_conv=strides_conv,
        pool_size=pool_size,
        strides_pool=pool_size,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        padding=padding,
        debug=debug
    )

    return model


def create_time_distributed_lstm_cnn(
        input_shape: Tuple[int, int, int],
        n_conv_layers: int,
        filters: int,
        filter_size: int,
        strides_conv: int,
        pool_size: int,
        strides_pool: int,
        lstm_units: int,
        dropout_rate: float,
        l2_reg: float,
        padding: str = 'same',
        debug: bool = False
) -> tf.keras.Model:
    """
    Creates a Time-Distributed CNN-LSTM model for processing EEG signal segments.

    This model architecture is designed to process EEG data that has been segmented into
    smaller time windows. It first applies a series of convolutional and pooling layers,
    wrapped in TimeDistributed layers to preserve the temporal structure, followed by
    an LSTM layer to capture sequential dependencies. The final output is a binary
    classification.

    :param input_shape: The shape of a single EEG segment (height, width, channels).
    :param n_conv_layers: The number of convolutional layers.
    :param filters: The number of filters in each convolutional layer.
    :param filter_size: The size of the filters in the convolutional layers.
    :param strides_conv: The stride length for the convolutional layers.
    :param pool_size: The size of the pooling window for the max-pooling layers.
    :param strides_pool: The stride length for the max-pooling layers.
    :param lstm_units: The number of units in the LSTM layer.
    :param dropout_rate: The dropout rate applied after the LSTM layer.
    :param l2_reg: The L2 regularization factor for the convolutional layers.
    :param padding: The type of padding to use in the convolutional and pooling layers ('same' or 'valid').
    :param debug: If True, prints the dimensions of the data after each layer for debugging purposes.
    :return: A compiled Keras Model ready for training.
    """
    if debug:
        print(f"Parameters: ")

    inputs = tf.keras.Input(shape=(None, *input_shape))
    x = inputs
    if debug:
        print(f"Input shape: {x.shape}")

    for i in range(n_conv_layers):
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters, (filter_size, filter_size), strides=strides_conv, padding=padding,
                                   activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)))(x)
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

# def create_model_experimental(
#         trial: optuna.Trial,
#         input_shape: Tuple[int, int, int],
#         debug: bool = False
# ) -> tf.keras.Model:
#     """
#     Creates a time-distributed LSTM-CNN model based on hyperparameters suggested by Optuna.
#
#     This function uses the `trial` object from Optuna to suggest hyperparameters for the convolutional layers,
#     pooling layers, and the LSTM layer. It also ensures that the dimensions after each convolution and pooling
#     operation are valid. If dimensions become invalid, the trial is pruned.
#
#     :param trial: The Optuna trial object used to suggest hyperparameters.
#     :param input_shape: The shape of the input data, specified as (height, width, channels).
#     :param debug: A flag to indicate if debug mode is enabled. Defaults to False.
#     :return: A compiled Keras model with the suggested hyperparameters.
#     :raises optuna.exceptions.TrialPruned: If the dimensions after convolution or pooling become invalid.
#     """
#     n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
#     filters_list, filter_size_list, strides_conv_list = [], [], []
#     pool_size_list, strides_pool_list = [], []
#
#     height, width = input_shape[0], input_shape[1]
#
#     for i in range(n_conv_layers):
#         filters = trial.suggest_int(f'filters_{i + 1}', 16, 128, step=16)
#
#         max_filter_size = min(height, width) // 2
#         if max_filter_size < 2:
#             raise optuna.exceptions.TrialPruned(
#                 f"Filter size is too large for the current dimensions at layer {i + 1}.")
#         filter_size = trial.suggest_int(f'filter_size_{i + 1}', 2, max_filter_size)
#         strides_conv = trial.suggest_int(f'strides_conv_{i + 1}', 1, 2)
#
#         height = (height - filter_size) // strides_conv + 1
#         width = (width - filter_size) // strides_conv + 1
#
#         if height <= 0 or width <= 0:
#             raise optuna.exceptions.TrialPruned(f"Invalid dimensions after Conv2D at layer {i + 1}.")
#
#         pool_size = trial.suggest_int(f'pool_size_{i + 1}', 2, 3)
#         strides_pool = trial.suggest_int(f'strides_pool_{i + 1}', 1, 2)
#
#         height = (height - pool_size) // strides_pool + 1
#         width = (width - pool_size) // strides_pool + 1
#
#         if height <= 0 or width <= 0:
#             raise optuna.exceptions.TrialPruned(f"Invalid dimensions after MaxPooling at layer {i + 1}.")
#
#         filters_list.append(filters)
#         filter_size_list.append(filter_size)
#         strides_conv_list.append(strides_conv)
#         pool_size_list.append(pool_size)
#         strides_pool_list.append(strides_pool)
#
#     lstm_units = trial.suggest_int('lstm_units', 64, 256, step=64)
#     dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
#     l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
#
#     model = create_time_distributed_lstm_cnn(
#         input_shape=input_shape,
#         n_conv_layers=n_conv_layers,
#         filters_list=filters_list,
#         filter_size_list=filter_size_list,
#         strides_conv_list=strides_conv_list,
#         pool_size_list=pool_size_list,
#         strides_pool_list=strides_pool_list,
#         lstm_units=lstm_units,
#         dropout_rate=dropout_rate,
#         l2_reg=l2_reg,
#         debug=debug
#     )
#
#     return model


# def create_time_distributed_lstm_cnn_experimental(input_shape: Tuple[int, int, int],
#                                      n_conv_layers: int,
#                                      filters_list: List[int],
#                                      filter_size_list: List[int],
#                                      strides_conv_list: List[int],
#                                      pool_size_list: List[int],
#                                      strides_pool_list: List[int],
#                                      lstm_units: int,
#                                      dropout_rate: float,
#                                      l2_reg: float,
#                                      padding: str = 'same',
#                                      debug: bool = False) -> tf.keras.Model:
#     """
#     Creates a Time-Distributed CNN-LSTM model for processing EEG signal segments.
#
#     This model architecture is designed to process EEG data that has been segmented into
#     smaller time windows. It first applies a series of convolutional and pooling layers,
#     wrapped in TimeDistributed layers to preserve the temporal structure, followed by
#     an LSTM layer to capture sequential dependencies. The final output is a binary
#     classification.
#
#     :param input_shape:
#         A tuple representing the shape of a single EEG segment (height, width, channels).
#     :param n_conv_layers:
#         The number of convolutional layers in the model.
#     :param filters_list:
#         A list of integers where each integer represents the number of filters in the corresponding convolutional layer.
#     :param filter_size_list:
#         A list of integers where each integer represents the filter size for the corresponding convolutional layer.
#     :param strides_conv_list:
#         A list of integers where each integer represents the stride length for the corresponding convolutional layer.
#     :param pool_size_list:
#         A list of integers where each integer represents the pool size for the corresponding max-pooling layer.
#     :param strides_pool_list:
#         A list of integers where each integer represents the stride length for the corresponding max-pooling layer.
#     :param lstm_units:
#         The number of units in the LSTM layer.
#     :param dropout_rate:
#         The dropout rate applied after the LSTM layer to prevent overfitting.
#     :param l2_reg:
#         The L2 regularization factor applied to the kernel weights of the convolutional layers.
#     :param debug:
#         A boolean flag to enable or disable debug print statements.
#     :return: A compiled Keras Model ready for training.
#     """
#     inputs = tf.keras.Input(shape=(None, *input_shape))
#     x = inputs
#
#     if debug:
#         print(f"Input shape: {x.shape}")
#
#     for i in range(n_conv_layers):
#         filters = filters_list[i]
#         filter_size = filter_size_list[i]
#         strides_conv = strides_conv_list[i]
#         pool_size = pool_size_list[i]
#         strides_pool = strides_pool_list[i]
#
#         x = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.Conv2D(filters, (filter_size, filter_size), strides=strides_conv, padding=padding,
#                                    activation='relu', kernel_regularizer=l2(l2_reg)))(x)
#         x = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.MaxPooling2D((pool_size, pool_size), strides=strides_pool))(x)
#
#         if debug:
#             print(f"After Conv2D layer {i + 1}, shape: {x.shape}")
#
#     x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
#     if debug:
#         print(f"After Flatten, shape: {x.shape}")
#
#     x = tf.keras.layers.LSTM(lstm_units)(x)
#     x = tf.keras.layers.Dropout(dropout_rate)(x)
#     if debug:
#         print(f"After LSTM, shape: {x.shape}")
#
#     outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model
