import numpy as np
import pandas as pd
from typing import List
from scipy import signal


def read_eeg_data(file_path: str) -> List[np.ndarray]:
    """
    Reads EEG data from a file and splits it into individual channels.

    :param file_path: The path to the EEG data file.
    :type file_path: str
    :return: A list of numpy arrays, each containing data from one EEG channel.
    :rtype: List[np.ndarray]

    The file is expected to contain data for 16 channels, with each channel
    having 7680 samples. The data is read into a Pandas DataFrame and then split
    into individual channel arrays.
    """
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    num_samples_per_channel = 7680
    num_channels = 16

    # Splitting data into individual channels
    channels = []
    for i in range(num_channels):
        start_index = i * num_samples_per_channel
        end_index = (i + 1) * num_samples_per_channel
        channel_data = df.iloc[start_index:end_index, 0].values
        channels.append(channel_data)
    return channels


def scale_minmax(X: np.ndarray, min: float = 0.0, max: float = 1.0) -> np.ndarray:
    """
    Scales the input array X to a specified range [min, max].

    :param X: The input array to be scaled.
    :type X: np.ndarray
    :param min: The minimum value of the scaled output range, defaults to 0.0.
    :type min: float, optional
    :param max: The maximum value of the scaled output range, defaults to 1.0.
    :type max: float, optional
    :return: The scaled array.
    :rtype: np.ndarray

    The function normalizes the input array to the range [0, 1], then scales
    it to the specified range [min, max].
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def convert_to_image(mh_distribution: np.ndarray, flip: bool = True) -> np.ndarray:
    """
    Converts a matrix of time-frequency data into an image.

    :param mh_distribution: The input time-frequency distribution matrix.
    :type mh_distribution: np.ndarray
    :param flip: Whether to vertically flip the image, defaults to True.
    :type flip: bool, optional
    :return: The resulting image as a numpy array of type uint8.
    :rtype: np.ndarray

    The function scales the time-frequency distribution matrix to the range
    [0, 255] and converts it to an 8-bit unsigned integer format (with colors inverted). 
    Optionally, the image can be flipped vertically and the.
    """
    img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)
    img = 255 - img  # invert to make black regions indicate more energy
    return img
