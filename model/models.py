from typing import Tuple

import optuna
import tensorflow as tf
from keras import layers
from tensorflow.keras.applications import EfficientNetB0


def model_cnn_lstm(
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
        if debug:
            print(f"After Pooling2D layer {i + 1}, shape: {x.shape}")

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((pool_size, pool_size), strides=strides_pool))(x)
        if debug:
            print(f"After Conv2D layer {i + 1}, shape: {x.shape}")

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    if debug:
        print(f"After Global Average Pooling, shape: {x.shape}")

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    if debug:
        print(f"After LSTM, shape: {x.shape}")

    x = tf.keras.layers.Dense(1, name='dense_logits')(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_cnn3d(
        input_shape: Tuple[int, int, int],
        n_conv_layers: int,
        filters: int,
        filter_size: int,
        strides_conv: int,
        pool_size: int,
        strides_pool: int,
        dropout_rate: float,
        l2_reg: float,
        padding: str = 'same',
        debug: bool = False
) -> tf.keras.Model:
    """
    Creates a Conv3D CNN model for processing EEG signal segments.

    This model architecture is designed to process EEG data that has been segmented into
    smaller time windows. It first applies a series of convolutional and pooling layers.
    The final output is a binary classification.

    :param input_shape: The shape of a single EEG segment (height, width, channels).
    :param n_conv_layers: The number of convolutional layers.
    :param filters: The number of filters in each convolutional layer.
    :param filter_size: The size of the filters in the convolutional layers.
    :param strides_conv: The stride length for the convolutional layers.
    :param pool_size: The size of the pooling window for the max-pooling layers.
    :param strides_pool: The stride length for the max-pooling layers.
    :param dropout_rate: The dropout rate.
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
        x = tf.keras.layers.Conv3D(filters, (filter_size, filter_size, filter_size), strides=strides_conv,
                                   padding=padding, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        if debug:
            print(f"After Conv3D layer {i + 1}, shape: {x.shape}")
        x = tf.keras.layers.MaxPooling3D((1, pool_size, pool_size), strides=(1, strides_pool, strides_pool))(x)
        if debug:
            print(f"After MaxPooling3D layer {i + 1}, shape: {x.shape}")
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    if debug:
        print(f"After Global Average 3D Pooling, shape: {x.shape}")

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(1, name='dense_logits')(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_cnn_lstm_prepared(
        input_shape: Tuple[int, int, int],
        lstm_units: int,
        dropout_rate: float = 0.5,
        merge_layer: str = 'avg_pool',
        merge_layer_lstm: str = 'pool2d',
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
    :param lstm_units: The number of units in the LSTM layer.
    :param dropout_rate: The dropout rate applied after the LSTM layer.
    :param debug: If True, prints the dimensions of the data after each layer for debugging purposes.
    :return: A compiled Keras Model ready for training.
    """
    if debug:
        print(f"Parameters: ")

    inputs = tf.keras.Input(shape=(None, *input_shape))
    x = inputs
    if debug:
        print(f"Input shape: {x.shape}")

    # Layer 1: Conv2D
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(96, (9, 9), strides=3, padding='valid', activation='relu'))(x)
    if debug:
        print(f"After Conv2D layer 1, shape: {x.shape}")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((3, 3), strides=2))(x)
    if debug:
        print(f"After MaxPooling2D layer 1, shape: {x.shape}")

    # Layer 2: Conv2D
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(256, (5, 5), strides=1, padding='valid', activation='relu'))(x)
    if debug:
        print(f"After Conv2D layer 2, shape: {x.shape}")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((3, 3), strides=2))(x)
    if debug:
        print(f"After MaxPooling2D layer 2, shape: {x.shape}")

    # Layer 3: Conv2D
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(384, (3, 3), strides=1, padding='valid', activation='relu'))(x)
    if debug:
        print(f"After Conv2D layer 3, shape: {x.shape}")

    # Layer 4: Conv2D
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(384, (3, 3), strides=1, padding='valid', activation='relu'))(x)
    if debug:
        print(f"After Conv2D layer 4, shape: {x.shape}")

    # Layer 5: Conv2D
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='valid', activation='relu'))(x)
    if debug:
        print(f"After Conv2D layer 5, shape: {x.shape}")

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=2))(x)
    if debug:
        print(f"After MaxPooling2D layer 3, shape: {x.shape}")

    if merge_layer == 'avg_pool':
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'))(x)
    elif merge_layer == 'max_pool':
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPooling2D(name='max_pool'))(x)
    elif merge_layer == 'flatten':
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(x)
    if debug:
        print(f"After Merging before Dense, shape: {x.shape}")

    # Dense Layers
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096, activation='relu'))(x)
    if debug:
        print(f"After Dense layer 1, shape: {x.shape}")
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096, activation='relu'))(x)
    if debug:
        print(f"After Dense layer 2, shape: {x.shape}")
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(192, activation='relu'))(x)
    if debug:
        print(f"After Dense layer 3, shape: {x.shape}")
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)

    # Flatten the features
    if merge_layer_lstm == 'pool2d':
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        if debug:
            print(f"After Global Average Pooling, shape: {x.shape}")
    elif merge_layer_lstm == 'flatten':
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        if debug:
            print(f"After Flatten, shape: {x.shape}")

    # LSTM Layer
    x = tf.keras.layers.LSTM(lstm_units)(x)
    if debug:
        print(f"After LSTM, shape: {x.shape}")

    x = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def model_cnn_prepared(
        input_shape: Tuple[int, int, int],
        trial=None,
        learning_rate=4.2666514538236414e-05,
        dropout_rate: float = 0.5,
        # filters: [int] = [96, 256, 384, 384, 256],
        # kernel_sizes: [int] = [9, 5, 3, 3, 3],
        # pool_sizes: [int] = [3, 3, 2],
        # dense_units: [int] = [4096, 4096, 192],
        merge_layer: str = 'avg_pool',
        padding: str = 'valid',
        metrics=None,
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
    :param dropout_rate: The dropout rate applied after the LSTM layer.
    :param debug: If True, prints the dimensions of the data after each layer for debugging purposes.
    :return: A compiled Keras Model ready for training.
    """
    if debug:
        print(f"Parameters: ")

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if debug:
        print(f"Input shape: {x.shape}")

    # Layer 1: Conv2D
    x = tf.keras.layers.Conv2D(160, (6, 6), strides=3, padding=padding,
                               activation='relu')(x)
    if debug:
        print(f"After Conv2D layer 1, shape: {x.shape}")

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    if debug:
        print(f"After MaxPooling2D layer 1, shape: {x.shape}")

    # Layer 2: Conv2D
    x = tf.keras.layers.Conv2D(160, (7, 7), strides=1, padding=padding,
                               activation='relu')(x)
    if debug:
        print(f"After Conv2D layer 2, shape: {x.shape}")

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    if debug:
        print(f"After MaxPooling2D layer 2, shape: {x.shape}")

    # Layer 3: Conv2D
    x = tf.keras.layers.Conv2D(160, (5, 5), strides=1, padding=padding,
                               activation='relu')(x)
    if debug:
        print(f"After Conv2D layer 3, shape: {x.shape}")

    # Layer 4: Conv2D
    x = tf.keras.layers.Conv2D(96, (3, 3), strides=1, padding=padding,
                               activation='relu')(x)
    if debug:
        print(f"After Conv2D layer 4, shape: {x.shape}")

    # Layer 5: Conv2D
    x = tf.keras.layers.Conv2D(288, (5, 5), strides=1, padding=padding,
                               activation='relu')(x)
    if debug:
        print(f"After Conv2D layer 5, shape: {x.shape}")

    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    if debug:
        print(f"After MaxPooling2D layer 3, shape: {x.shape}")

    if merge_layer == 'avg_pool':
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif merge_layer == 'max_pool':
        x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
    elif merge_layer == 'flatten':
        x = tf.keras.layers.Flatten(name='flatten')(x)
    if debug:
        print(f"After Merging before Dense, shape: {x.shape}")

    # Dense Layers
    x = tf.keras.layers.Dense(2692, activation='relu')(x)
    if debug:
        print(f"After Dense layer 1, shape: {x.shape}")
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(692, activation='relu')(x)
    if debug:
        print(f"After Dense layer 2, shape: {x.shape}")
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1942, activation='relu')(x)
    if debug:
        print(f"After Dense layer 3, shape: {x.shape}")
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output Layer
    x = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    if metrics is None:
        metrics = ['accuracy']
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def model_cnn_search(
        trial: optuna.Trial,
        input_shape: Tuple[int, int, int],
        learning_rate: float,
        metrics=None,
        debug: bool = False
) -> tf.keras.Model:
    padding = trial.suggest_categorical('padding', ['same', 'valid'])
    position_batchnorm = trial.suggest_categorical('position_batchnorm', ['before', 'after'])

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    num_conv_layers = trial.suggest_int('num_conv_layers', 2, 7)

    for i in range(num_conv_layers):
        filters = trial.suggest_int(f'filter_number_{i}', 32, 416, step=64)
        kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 9, step=1)
        use_pooling = trial.suggest_categorical(f'use_pooling_{i}', [True, False])
        pool_size = trial.suggest_int(f'pool_size_{i}', 2, 4, step=1)
        use_batchnorm = trial.suggest_categorical(f'use_batchnorm_conv_{i}', [True, False])
        use_dropout = trial.suggest_categorical(f'use_dropout_conv_{i}', [True, False])

        x = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), strides=1, padding=padding, activation=None)(x)

        if use_batchnorm and position_batchnorm == 'before':
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)

        if use_batchnorm and position_batchnorm == 'after':
            x = tf.keras.layers.BatchNormalization()(x)

        if use_pooling:
            x = tf.keras.layers.MaxPooling2D((pool_size, pool_size), strides=2)(x)

        if use_dropout:
            dropout_rate = trial.suggest_float(f'dropout_rate_conv_{i}', 0.2, 0.5, step=0.1)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if debug:
            print(f"After Conv2d layer {i}, shape: {x.shape}")

    # merge_layer = trial.suggest_categorical('merge_layer', ['avg_pool', 'flatten'])
    merge_layer = trial.suggest_categorical('merge_layer', ['avg_pool'])
    if merge_layer == 'avg_pool':
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif merge_layer == 'flatten':
        x = tf.keras.layers.Flatten(name='flatten')(x)
    if debug:
        print(f"After Merging before Dense, shape: {x.shape}")

    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)

    for i in range(num_dense_layers):
        # if merge_layer == 'avg_pool':
        #     dense_units = trial.suggest_int(f'dense_units_{i}', 32, 512, step=32)
        # else:
        dense_units = trial.suggest_int(f'dense_units_{i}', 32, 4064, step=32)
        use_batchnorm = trial.suggest_categorical(f'use_batchnorm_dense_{i}', [True, False])
        use_dropout = trial.suggest_categorical(f'use_dropout_dense_{i}', [True, False])

        x = tf.keras.layers.Dense(dense_units, activation=None)(x)

        if use_batchnorm and position_batchnorm == 'before':
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)

        if use_batchnorm and position_batchnorm == 'after':
            x = tf.keras.layers.BatchNormalization()(x)

        if use_dropout:
            dropout_rate = trial.suggest_float(f'dropout_rate_dense_{i}', 0.2, 0.5, step=0.1)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if debug:
            print(f"After Dense layer {i}, shape: {x.shape}")

    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    if debug:
        print(f"After last Dense layer, shape: {x.shape}")

    model = tf.keras.Model(inputs=inputs, outputs=x)
    # model.summary()
    num_params = model.count_params()
    print(f"Number of trainable parameters: {num_params}")
    if num_params > 12_000_000:
        raise optuna.exceptions.TrialPruned()

    if metrics is None:
        metrics = ['accuracy']
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)
    return model


def efficientnet(
        input_shape: Tuple[int, int, int],
        dropout_rate: float = 0.5,
        debug: bool = False
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    if input_shape[-1] == 3:
        data_format = 'channels_last'
    else:
        data_format = 'channels_first'
    x = layers.GlobalAveragePooling2D(name="avg_pool", data_format=data_format)(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = dropout_rate
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(2, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    return model
