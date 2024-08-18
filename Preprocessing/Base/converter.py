import pandas as pd
from mne.io import read_raw_edf
from typing import List, Dict, Hashable
from scipy.signal import resample, butter, filtfilt, get_window
from tftb.processing import MargenauHillDistribution
import numpy as np


def convert_dataframe_to_dict(dataframe: pd.DataFrame) -> Dict[Hashable, List]:
    """
    Helper function to map from dataframe to dict. Mapping column names to columns.

    :param dataframe: Dataframe with all signals
    :return: Dict mapping column names to columns.
    """
    signals_dict = {}

    for name, signal in dataframe.items():
        signals_dict[name] = signal.to_list()
    return signals_dict


def get_signals_from_csv(filename: str) -> pd.DataFrame:
    """
    Creates generator for csv file. File is split into 3072 rows chunks (one trial with one condition).
    Generator is preferred here so the RAM doesn't go boom :)

    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: DataFrame with signals
    """
    csv_labels = list(pd.read_csv("columnLabels.csv").columns)
    for df in pd.read_csv(filename, names=csv_labels, chunksize=3072):
        yield df.iloc[:, 4:-6]


def get_signals_from_eea(filename: str, measurements_per_channel: int = 7680,
                         channels: List | None = None) -> pd.DataFrame:
    """
    Reads eea file and split it into chunks and map it to channels.

    :param channels: List of channels.
    :param measurements_per_channel: How many measurement are per channel. Default is 7680 (for purpose of our dataset)
    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: DataFrame with signals
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        if channels is None:
            channels = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        data = {channels[k]: lines[i:i + measurements_per_channel] for k, i in
                enumerate(range(0, len(lines), measurements_per_channel))}
        return pd.DataFrame(data, dtype=float)


def get_signals_from_edf(filename: str) -> pd.DataFrame:
    """
    Reads measurements from edf file and map it to DataFrame

    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: DataFrame with signals
    """
    edf_data = read_raw_edf(filename)
    channels = edf_data.ch_names
    data, times = edf_data[:]

    pd_data = {channels[i]: data[i] for i in range(len(channels))}

    return pd.DataFrame(pd_data)


def lowpass_filter(signals: pd.DataFrame, original_freq_sampling: int, cutoff_freq: int = 64) -> pd.DataFrame:
    """
    Performs lowpass filter on all signals.

    :param original_freq_sampling: Original frequency of sampling
    :param signals: Dataframe with all signals in dataset
    :param cutoff_freq: Frequency from which we want to cutoff frequencies
    :return: Dataframe with applied lowpass filter
    """
    signals_dict = convert_dataframe_to_dict(signals)
    max_freq = original_freq_sampling / 2

    normal_cutoff = cutoff_freq / max_freq

    b, a = butter(4, normal_cutoff, btype='low', analog=False)

    filtered = {k: filtfilt(b, a, v) for k, v in signals_dict.items()}

    return pd.DataFrame(filtered)


def resample_signal(signals: pd.DataFrame, original_sample_rate: int, new_sample_rate: int = 128) -> pd.DataFrame:
    def resample_list(signal_list: List[float]) -> List[float]:
        sample_num = int(len(signal_list) * new_sample_rate / original_sample_rate)
        return resample(signal_list, sample_num)

    signals_dict = convert_dataframe_to_dict(signals)

    resampled = {k: resample_list(v) for k, v in signals_dict.items()}

    return pd.DataFrame(resampled)


def split_into_time_windows(signal: List[float] | pd.Series, sample_frequency: int, secs: float = 3) -> pd.Series | \
                                                                                                        List[float]:
    """
    Returns some time window from signal. It's generator for simplicity sake.
    TODO: currently chops of last few secs if data cannot be split into perfect chunks of secs

    :param signal: Values from one channel.
    :param sample_frequency: Frequency at which data is sampled.
    :param secs: By what time divide data
    """

    time = len(signal) / sample_frequency

    chunks = int(time // 3)

    for i in range(chunks - 1):
        # [chunk_start : chunk_end]
        yield signal[(secs * sample_frequency) * i: (secs * sample_frequency) * (i + 1)]


def calculate_margenau_lib(signal: List[float]) -> List[List[float]]:
    """
    Calculate MH distribution using TFTB package

    :param signal:
    :return:
    """
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    return tfr_real.tfr


def calculate_margenau_hubert(signal, window='hann', nfft=1024, scaling_factor=10):
    """
    Calculate MH distribution using Hubert's algorithm.

    :param signal:
    :param window:
    :param nfft:
    :param scaling_factor:
    :return:
    """
    n = len(signal)
    win = get_window(window, n)
    signal_win = signal * win

    step = nfft // 2
    tfr = np.zeros((n // step, nfft // scaling_factor), dtype=complex)
    for i in range(0, n - step, step):
        segment = signal_win[i:i + nfft]
        if len(segment) < nfft:
            segment = np.pad(segment, (0, nfft - len(segment)), 'constant')
        for j in range(nfft // scaling_factor):
            k = np.arange(nfft)
            k = k[(k >= j * scaling_factor) & (k < (2 * nfft - j * scaling_factor))]
            tfr[i // step, j] = np.sum(segment[k] * np.conj(segment[k - j * scaling_factor]))

    tfr = np.real(np.fft.fft(tfr, axis=0))
    tfr = np.abs(tfr)
    # tfr = tfr / np.max(tfr)
    return tfr


