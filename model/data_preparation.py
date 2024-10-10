import random
from typing import List, Dict, Any, Tuple
import numpy as np
import tensorflow as tf
import joblib
import cv2
from tensorflow.keras.utils import to_categorical

# TODO: Replace the mhd function
from model.mhd_temp import margenau_hill_distribution as mhd
from model.params import SEGMENT_SIZE_SEC, SAMPLING_RATE, COMMON_CHANNELS, IMAGE_SIZE, \
    DATA_SAMPLE_SHAPE, SEGMENTS_SPLIT, SEGMENT


def load_eeg_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads EEG data from a file using joblib.

    :param filepath: Path to the file containing the EEG data.
    :return: A list of dictionaries, where each dictionary contains an EEG recording.
    """
    with open(filepath, 'rb') as f:
        return joblib.load(f)


def prepare_eeg_data(eeg_data: List[Dict[str, Any]], segment_size: int = SEGMENT_SIZE_SEC,
                     sampling_rate: int = SAMPLING_RATE, channel_list: [str] = COMMON_CHANNELS,
                     label: int = 0, segment: bool = False) -> List[Dict[str, Any]]:
    """
    Prepares EEG data for further analysis by filtering and optionally segmenting the data.

    :param eeg_data: List of dictionaries containing EEG data. Each dictionary should have a key 'eeg'
                     with another dictionary mapping channel names to their respective signal values.
    :param segment_size: Size of each segment in seconds, used if segmentation is required.
    :param sampling_rate: Sampling rate of the EEG data in Hz.
    :param channel_list: List of channel names to filter the data by.
    :param label: Label to assign to each sample, typically 0 for healthy and 1 for ill.
    :param segment: Boolean flag indicating whether to segment the signal.
    :returns: List of dictionaries with keys 'data' containing the processed EEG signals and 'label'
              containing the corresponding labels.
    """
    data = []
    for sample in eeg_data:
        filtered_channel_data = np.array(
            [sample['eeg'][channel] for channel in channel_list if channel in sample['eeg']])
        if segment:
            filtered_channel_data = segment_signal(filtered_channel_data, segment_size, sampling_rate)
        data.append({'data': filtered_channel_data, 'label': np.float32(label)})  # Add label from arg
    return data


def segment_signal(signal: np.ndarray, segment_size: int, sampling_rate: int) -> np.ndarray:
    """
    Segments an EEG signal into smaller segments.

    :param signal: 2D array representing EEG data, with shape (channels, time points).
    :param segment_size: Size of each segment in seconds.
    :param sampling_rate: Sampling rate of the EEG data in Hz.
    :return: A 3D array where each element represents a segment with shape (channels, segment_length).
    """
    signal = np.asarray(signal)
    segment_length = segment_size * sampling_rate
    segments = [signal[:, i:i + segment_length] for i in range(0, signal.shape[1] - segment_length + 1, segment_length)]
    return np.array(segments)


def get_data(health_path, ill_path, segment=SEGMENT):
    print("Data loading and preparation")
    health_data = load_eeg_data(health_path)
    ill_data = load_eeg_data(ill_path)

    # Data preparation
    train_data_schizophrenia = prepare_eeg_data(ill_data, label=1, segment=segment)
    train_data_health = prepare_eeg_data(health_data, label=0, segment=segment)

    signals_data = train_data_schizophrenia + train_data_health
    np.random.shuffle(signals_data)

    # Signals to M-H distribution
    print("Signals to M-H distribution")
    for idx, signal in enumerate(signals_data):
        print(f"Processing: {idx}/{len(signals_data)}")
        if segment:
            signal["data"] = preprocess_eeg_segmented_sample(signal["data"])
        else:
            signal["data"] = preprocess_eeg_sample(signal["data"])

    return signals_data


def preprocess_eeg_sample(sample: List[np.ndarray], size: Tuple[int, int] = IMAGE_SIZE,
                          norm_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Converts each channel in an EEG sample to an image representation (M-H distribution).

    :param sample: List of numpy arrays representing EEG channels.
    :param size: Tuple representing the sizes to which each channel's image representation is resized.
    :param norm_range: Tuple indicating the normalization range for the signal channel (image representation). Default is (0, 1).
    :return: A numpy array with the M-H distribution images stacked along the depth dimension.
    """
    return np.stack([signal_to_mhd_image(channel, size=size, range=norm_range) for channel in sample], axis=-1)


def preprocess_eeg_segmented_sample(sample: [np.ndarray], size: Tuple[int, int] = IMAGE_SIZE,
                                    norm_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Processes EEG sample by converting each segment into an image representation (M-H distribution).

    Each channel in a segment is converted into an image, and the channels are stacked
    along the depth dimension, creating an 'image' for each segment. The function
    returns an array of these 'images', along with the corresponding label.

    :param sample: A numpy array of shape (n_segments, n_channels, segment_length), where:
            - n_segments: The number of segments in the sample.
            - n_channels: The number of EEG channels.
            - segment_length: The length of each segment.
    :param size: Tuple representing the sizes to which each channel's image representation is resized.
    :param norm_range: Tuple indicating the normalization range for the signal channel (image representation). Default is (0, 1).
    :return: A numpy array of shape (n_segments, segment_height, segment_width, n_channels),
          representing the processed segments as 'images'.

    Note:
        - segment_height is equal to segment_length/2 (mirror frequencies are cropped).

    Example:
        If `sample` has a shape of (20, 16, 384), the returned array will have
        a shape of (20, 384/2, 384, 16), assuming each channel's signal segment is converted to a
        (384/2)x(384) image.
    """
    processed_segments = []
    for segment in sample:
        segment_image = preprocess_eeg_sample(segment, size, norm_range)
        processed_segments.append(segment_image)
    return np.array(processed_segments)


def signal_to_mhd_image(signal: np.ndarray, size: tuple = None, range: tuple = None) -> np.ndarray:
    """
    Converts a 1D signal into a 2D image using the Margenau-Hill distribution.

    The signal is transformed into an image representation using the
    Margenau-Hill distribution, scaled to a specified size and range.

    :param signal: A 1D numpy array representing the EEG signal for a single channel.
    :param size: A tuple specifying the desired size (height, width) of the output image.
                 Default is (N/2 x N), where N is len of the signal.
    :param range: A tuple specifying the desired range (min, max) of the output image values.
    :return: A 2D numpy array of the specified size, representing the signal as an image.

    Note:
        - Mirror frequencies are cropped (N/2, N)
    """
    mh_distribution, ts = mhd(signal)
    mh_distribution = abs(mh_distribution)
    mh_distribution = mh_distribution[:(mh_distribution.shape[0] // 2), :]

    if size:
        mh_distribution = cv2.resize(mh_distribution, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    if range:
        mh_distribution = scale_minmax(mh_distribution, range[0], range[1])
    return mh_distribution


def scale_minmax(X: np.ndarray, min: float = 0.0, max: float = 1.0) -> np.ndarray:
    """
    Scales a numpy array to a specified range [min, max].

    :param X: A numpy array to be scaled.
    :param min: The minimum value of the scaled array. Default is 0.0.
    :param max: The maximum value of the scaled array. Default is 1.0.
    :return: A numpy array with values scaled to the range [min, max].
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def create_eeg_dataset(data: List[Dict[str, np.ndarray]], batch_size: int, shuffle: bool = False,
                       repeat: bool = True, segment_split: bool = SEGMENTS_SPLIT,
                       data_shape=DATA_SAMPLE_SHAPE) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from EEG data,
    where data contains already segmented signals and preprocessed into M-H distributions.

    This function generates a dataset that yields batches of EEG data segments
    and their corresponding labels. It optionally shuffles the data and repeats
    it for multiple epochs.

    :param data: A list of dictionaries, each containing:
        - 'segments': A numpy array of shape (n_segments, segment_height, segment_width, n_channels) or (segment_height, segment_width, n_channels).
        - 'label': An integer label associated with the data.
    :param batch_size: The number of samples per batch.
    :param shuffle: Whether to shuffle the data before batching. Default is False.
    :param repeat: Whether to repeat the dataset for multiple epochs. Default is True.
    :param segment_split: Whether to split the data into individual segments.
    :param data_shape: The shape of the data samples.
    :return: A `tf.data.Dataset` object that yields batches of data.

    The dataset's output signature:
        - The features have a shape of (None, segment_height, segment_width, n_channels) or (segment_height, segment_width, n_channels).
        - The labels are one-hot encoded with a shape of (2,).

    Note:
        If `shuffle` is True, the dataset will be shuffled with a buffer size of 100.
    """
    all_segments = []
    for sample in data:
        if segment_split:
            for segment in sample["segments"]:
                all_segments.append((segment, sample["label"]))
        else:
            all_segments.append((sample["segments"], sample["label"]))
    if shuffle:
        random.shuffle(all_segments)

    def generator():
        for segment, label in all_segments:
            yield segment, to_categorical(label, num_classes=2)

    output_signature = (
        tf.TensorSpec(shape=data_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
