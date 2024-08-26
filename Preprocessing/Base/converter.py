import math

import mne
import pandas as pd
from mne.io import read_raw_edf
from typing import List, Dict, Hashable

from mne.io.edf.edf import RawEDF
from scipy.signal import resample, butter, filtfilt, get_window, resample_poly, decimate
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


def get_signals_from_csv(filename: str, sample_frequency: int = 1024) -> mne.io.RawArray:
    """
    Creates generator for csv file. File is split into 3072 rows chunks (one trial with one condition).
    Generator is preferred here so the RAM doesn't go boom :)

    :param sample_frequency: Frequency of the dataset sampling
    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawArray from mne with signals
    """
    csv_labels = list(pd.read_csv("columnLabels.csv").columns)[4:-6]
    for df in pd.read_csv(filename, names=csv_labels, chunksize=3072):
        relevant_data = df.iloc[:, 4:-6]
        eeg_data = relevant_data.values.T
        channel_types = ['eeg'] * len(eeg_data)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=csv_labels)
        raw_data = mne.io.RawArray(eeg_data, info)
        yield raw_data


def get_signals_from_eea(filename: str, measurements_per_channel: int = 7680,
                         channels: List | None = None, sample_frequency=250) -> mne.io.RawArray:
    """
    Reads eea file and split it into chunks and map it to channels.

    :param sample_frequency: Frequency at which dataset was sampled
    :param channels: List of channels.
    :param measurements_per_channel: How many measurement are per channel. Default is 7680 (for purpose of our dataset)
    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawArray from mne with signals
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        if channels is None:
            channels = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        eeg_data = [lines[i:i + measurements_per_channel] for i in
                    range(0, len(lines), measurements_per_channel)]

        channel_types = ['eeg'] * len(channels)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=channels)
        raw_data = mne.io.RawArray(eeg_data, info)
        return raw_data


def get_signals_from_edf(filename: str) -> RawEDF:
    """
    Reads measurements from edf file and map it to DataFrame

    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawEDF with signals
    """
    edf_data = read_raw_edf(filename)
    return edf_data


def filter_mne(signals: mne.io.RawArray | RawEDF, cutoff_freq: int = 64) -> mne.io.RawArray | RawEDF:
    """
    Performs lowpass filter on all signals.

    :param signals: Dataframe with all signals in dataset
    :param cutoff_freq: Frequency from which we want to cutoff frequencies
    :return: Mne object with applied filter, object is the same that is passed
    """
    signals.filter(1, cutoff_freq)
    return signals


def resample_signal(signals: mne.io.RawArray | RawEDF, original_sample_rate: int,
                    new_sample_rate: int = 128) -> mne.io.RawArray:
    """
    Resample signal using decimate/resample_poly functions


    :param signals: Mne object with signals
    :param original_sample_rate: Sample rate of the original signal
    :param new_sample_rate: New sample rate of which we want make
    :return: Mne object with resampled signals
    """

    def check_if_power_of_two() -> bool:
        result = math.log(original_sample_rate, 2)

        result_int = int(result)

        # if numbers are not the same it means that in log function is reminder
        # which means that number is not the power of two
        return result == result_int

    def resample_list(signal_list: List[float]):
        """
        Resample signal list using decimate/resample_poly

        :param signal_list: Signal from one channel. Param should be a list of floats.
        """
        should_use_poly = check_if_power_of_two()
        if should_use_poly:
            gcd = np.gcd(original_sample_rate, new_sample_rate)

            # Upsampling and downsampling factors
            up = original_sample_rate // gcd
            down = new_sample_rate // gcd

            # Apply low-pass FIR filter and perform resampling using resample_poly
            return resample_poly(signal_list, up, down)
        else:
            gcd = np.gcd(original_sample_rate, new_sample_rate)
            down = new_sample_rate // gcd
            # Apply low-pass FIR filter and perform resampling using resample_poly
            return decimate(signal_list, down)

    # function returns data | timestamps, timestamps are omitted (not used I guess)
    eeg_data_list, _ = signals.get_data()
    channel_names = signals.ch_names

    placeholder_data = []
    for eeg_data in eeg_data_list:
        placeholder_data.append(resample_list(eeg_data))

    channel_types = ['eeg'] * len(channel_names)

    info = mne.create_info(ch_types=channel_types, sfreq=new_sample_rate, ch_names=channel_names)
    raw_data = mne.io.RawArray(placeholder_data, info)
    return raw_data


def split_into_time_windows(signal: mne.io.RawArray, sample_frequency: int, secs: float = 3) -> pd.Series | \
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

    data, times = signal.get_data()

    for each_channel in data:
        for i in range(chunks - 1):
            # [chunk_start : chunk_end]
            yield each_channel[(secs * sample_frequency) * i: (secs * sample_frequency) * (i + 1)]


def calculate_margenau_lib(signal: List[float]) -> List[List[float]]:
    """
    Calculate MH distribution using TFTB package

    :param signal:
    :return:
    """
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    return tfr_real.tfr
