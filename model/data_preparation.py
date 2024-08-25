from typing import List, Dict, Any, Tuple
import numpy as np
import tensorflow as tf
import joblib

# TODO: Replace the mhd function
from mhd_temp import margenau_hill_distribution_spectrogram_tfrmhs_ifft as mhd
from params import SEGMENT_SIZE_SEC, SAMPLING_RATE, SEGMENT_COLUMNS, SEGMENT_ROWS, CHANNEL_NUMBER


def load_eeg_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads EEG data from a file using joblib.

    :param filepath: Path to the file containing the EEG data.
    :return: A list of dictionaries, where each dictionary contains an EEG recording.
    """
    with open(filepath, 'rb') as f:
        return joblib.load(f)


def load_and_segment_eeg_data(filepaths: List[str], segment_size: int = SEGMENT_SIZE_SEC,
                              sampling_rate: int = SAMPLING_RATE, label: int = 0) -> List[Dict[str, Any]]:
    """
    Loads EEG data from multiple files and segments each EEG recording.

    :param filepaths: List of paths to the files containing EEG data.
    :param segment_size: Size of each segment in seconds.
    :param sampling_rate: Sampling rate of the EEG data in Hz.
    :param label: Label to assign to all segments from the given files.
    :return: A list of dictionaries where each dictionary contains segmented EEG data and its corresponding label.
    """
    segmented_data = []
    for filepath in filepaths:
        eeg_data = load_eeg_data(filepath)
        for sample in eeg_data:
            segments = segment_signal(sample['eeg'], segment_size, sampling_rate)
            segmented_data.append({'segments': segments, 'label': label})  # dodajemy etykietę z argumentu
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


def preprocess_eeg_sample(sample: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Processes EEG sample by converting each segment into a 3D image representation.

    Each channel in a segment is converted into an image, and the channels are stacked
    along the depth dimension, creating a 3D image for each segment. The function
    returns an array of these 3D images, along with the corresponding label.

    :param sample: A dictionary containing:
        - 'segments': A numpy array of shape (n_segments, n_channels, segment_length), where:
            - n_segments: The number of segments in the sample.
            - n_channels: The number of EEG channels.
            - segment_length: The length of each segment.
        - 'label': An integer label associated with the sample.
    :return: A tuple containing:
        - A numpy array of shape (n_segments, segment_height, segment_width, n_channels),
          representing the processed segments as '3D images'.
        - An integer label corresponding to the sample.

    Example:
        If `sample['segments']` has a shape of (20, 16, 384), the returned array will have
        a shape of (20, 384, 384, 16), assuming each channel's signal is converted to a
        384x384 image.
    """
    segments, label = sample['segments'], sample['label']

    # Process each segment
    processed_segments = []
    for segment in segments:
        # Create N-channel image from the segment (one 1-channel image per channel)
        segment_image = np.stack([signal_to_image(channel) for channel in segment], axis=-1)
        processed_segments.append(segment_image)

    return np.array(processed_segments), label


def signal_to_image(signal: np.ndarray) -> np.ndarray:
    """
    Converts a 1D signal into a 2D image using the Margenau-Hill distribution.

    The signal is transformed into an image representation using the
    Margenau-Hill distribution, scaled to a range of 0-1.

    :param signal: A 1D numpy array representing the EEG signal for a single channel.
    :return: A 2D numpy array of type uint8, representing the signal as an image.

    Note:
        - Mirror frequencies are cropped (N/2, N)
    """
    mh_distribution, ts = mhd(signal)
    mh_distribution = abs(mh_distribution)
    mh_distribution = mh_distribution[:(mh_distribution.shape[0] // 2), :]

    img = scale_minmax(mh_distribution, 0, 1)
    # I think that further conversion is not needed since we do not use pretreated image models
    # img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    # img = np.flip(img, axis=0)
    # img = 255 - img
    return img


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
    Creates a TensorFlow dataset from EEG data.

    This function generates a dataset that yields batches of EEG data segments
    and their corresponding labels. It optionally shuffles the data and repeats
    it for multiple epochs.

    :param data: A list of dictionaries, each containing:
        - 'segments': A numpy array of shape (n_segments, n_channels, segment_length).
        - 'label': An integer label associated with the data.
    :param batch_size: The number of samples per batch.
    :param shuffle: Whether to shuffle the data before batching. Default is False.
    :param repeat: Whether to repeat the dataset for multiple epochs. Default is True.
    :return: A `tf.data.Dataset` object that yields batches of data.

    The dataset's output signature:
        - The features have a shape of (None, SEGMENT_WIDTH, SEGMENT_HEIGHT, CHANNEL_NUMBER).
        - The labels are integers.

    Note:
        If `shuffle` is True, the dataset will be shuffled with a buffer size of 10.
    """

    def generator():
        for sample in data:
            frames, label = preprocess_eeg_sample(sample)
            yield frames, label

    output_signature = (
        tf.TensorSpec(shape=(None, SEGMENT_ROWS, SEGMENT_COLUMNS, CHANNEL_NUMBER), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
