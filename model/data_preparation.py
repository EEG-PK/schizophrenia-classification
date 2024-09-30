from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
import joblib
import cv2

# TODO: Replace the mhd function
from mhd_temp import margenau_hill_distribution as mhd
from params import SEGMENT_SIZE_SEC, SAMPLING_RATE, SEGMENT_COLUMNS, SEGMENT_ROWS, DATASETS_DIR, \
    DATASET_DIR, SCHIZO_DUMP_FILE, HEALTH_DUMP_FILE, COMMON_CHANNELS


def load_eeg_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads EEG data from a file using joblib.

    :param filepath: Path to the file containing the EEG data.
    :return: A list of dictionaries, where each dictionary contains an EEG recording.
    """
    with open(filepath, 'rb') as f:
        return joblib.load(f)


def load_and_segment_eeg_data(filepaths: List[str], segment_size: int = SEGMENT_SIZE_SEC,
                              sampling_rate: int = SAMPLING_RATE, channel_list: [str] = COMMON_CHANNELS,
                              label: int = 0) -> List[Dict[str, Any]]:
    """
    Loads EEG data from multiple files and segments each EEG recording.
    :param filepaths: List of paths to the files containing EEG data.
    :param segment_size: Size of each segment in seconds.
    :param sampling_rate: Sampling rate of the EEG data in Hz.
    :param channel_list: Names of the EEG channels.
    :param label: Label to assign to all segments from the given files.
    :return: A list of dictionaries where each dictionary contains segmented EEG data and its corresponding label.
    """
    segmented_data = []
    for filepath in filepaths:
        eeg_data = load_eeg_data(filepath)
        for sample in eeg_data:
            channels = np.array([sample['eeg'][channel] for channel in channel_list if channel in sample['eeg']])
            segments = segment_signal(channels, segment_size, sampling_rate)
            segmented_data.append({'segments': segments, 'label': np.float32(label)})  # Add label from arg
    return segmented_data


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


def get_data():
    print("Data loading and segmentation")
    train_files_schizophrenia = [f"{DATASETS_DIR}/{DATASET_DIR}/{SCHIZO_DUMP_FILE}"]
    train_files_health = [f"{DATASETS_DIR}/{DATASET_DIR}/{HEALTH_DUMP_FILE}"]

    # Data segmentation
    segmented_train_data_schizophrenia = load_and_segment_eeg_data(train_files_schizophrenia, label=1)
    segmented_train_data_health = load_and_segment_eeg_data(train_files_health, label=0)
    segmented_signals_data = segmented_train_data_schizophrenia + segmented_train_data_health
    np.random.shuffle(segmented_signals_data)

    # Segments to M-H distribution
    print("Segments to M-H distribution")
    for signal in segmented_signals_data:
        signal["segments"] = preprocess_eeg_sample(signal["segments"])

    return segmented_signals_data


def preprocess_eeg_sample(sample: [np.ndarray]) -> np.ndarray:
    """
    Processes EEG sample by converting each segment into an image representation (M-H distribution).

    Each channel in a segment is converted into an image, and the channels are stacked
    along the depth dimension, creating an 'image' for each segment. The function
    returns an array of these 'images', along with the corresponding label.

    :param sample: A numpy array of shape (n_segments, n_channels, segment_length), where:
            - n_segments: The number of segments in the sample.
            - n_channels: The number of EEG channels.
            - segment_length: The length of each segment.
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
        segment_image = np.stack([signal_to_mhd_image(channel, range=(0, 1)) for channel in segment], axis=-1)
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
                       repeat: bool = True) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from EEG data,
    where data contains already segmented signals and preprocessed into M-H distributions.

    This function generates a dataset that yields batches of EEG data segments
    and their corresponding labels. It optionally shuffles the data and repeats
    it for multiple epochs.

    :param data: A list of dictionaries, each containing:
        - 'segments': A numpy array of shape (n_segments, segment_height, segment_width, n_channels).
        - 'label': An integer label associated with the data.
    :param batch_size: The number of samples per batch.
    :param shuffle: Whether to shuffle the data before batching. Default is False.
    :param repeat: Whether to repeat the dataset for multiple epochs. Default is True.
    :return: A `tf.data.Dataset` object that yields batches of data.

    The dataset's output signature:
        - The features have a shape of (None, N/2, N, CHANNEL_NUMBER), where N is len of the signal.
        - The labels are floats.

    Note:
        If `shuffle` is True, the dataset will be shuffled with a buffer size of 10.
    """

    def generator():
        for sample in data:
            yield sample["segments"], np.expand_dims(sample["label"], axis=-1)

    output_signature = (
        tf.TensorSpec(shape=(None, SEGMENT_ROWS, SEGMENT_COLUMNS, len(COMMON_CHANNELS)), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
