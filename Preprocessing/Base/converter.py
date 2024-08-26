import math

import mne
import pandas as pd
from mne.io import read_raw_edf
from typing import List, Literal

from mne.io.edf.edf import RawEDF
from scipy.signal import resample_poly, decimate
from tftb.processing import MargenauHillDistribution
import numpy as np
from joblib import dump
import os

common_channels = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz']


def get_signals_from_csv(filename: str, sample_frequency: int = 1024) -> mne.io.RawArray:
    """
    Creates generator for csv file. File is split into 3072 rows chunks (one trial with one condition).
    Generator is preferred here so the RAM doesn't go boom :)

    :param sample_frequency: Frequency of the dataset sampling
    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawArray from mne with signals
    """
    csv_labels = list(pd.read_csv("columnLabels.csv").columns)
    df: pd.DataFrame
    for df in pd.read_csv(filename, names=csv_labels, chunksize=3072):
        # if condition is other than 2 (passive listing to tone) skip the data
        # FIXME: change number 1 to 2
        if df.loc[df['condition'] != 1].size != 0:
            continue
        # get only columns which are in common channels in all datasets
        relevant_columns = df.columns.intersection(common_channels)
        relevant_data = df[relevant_columns]
        eeg_data = relevant_data.values.T
        channel_types = ['eeg'] * len(eeg_data)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=list(relevant_data.columns))
        raw_data = mne.io.RawArray(eeg_data, info)
        yield raw_data


# TODO: implement
def create_reference_electrode(signals: mne.io.RawArray | RawEDF) -> mne.io.RawArray | RawEDF:
    forward = mne.make_forward_solution(
        signals.info,
        # TODO: read about transposition of this
        trans=None,
    )
    eeg_data = mne.set_eeg_reference(signals, ref_channels='REST', forward=mne.make_forward_solution())
    return eeg_data


def get_signals_from_eea(filename: str, measurements_per_channel: int = 7680,
                         channels: List | None = None, sample_frequency: int = 250) -> mne.io.RawArray:
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
            # Order of this channel names are important, don't change it
            channels_original = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6',
                                 'O1', 'O2']
            # get a filtered channel names which are common in all datasets
            channels_filtered = [channels_original[i] for i in channels_original if
                                 channels_original[i] in common_channels]

        # we need to iterate through all values and skip those which are not common for all datasets
        eeg_data = [lines[i:i + measurements_per_channel] for i in
                    range(0, len(lines), measurements_per_channel) if channels_original[i] in common_channels]

        channel_types = ['eeg'] * len(channels)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=channels_filtered)
        raw_data = mne.io.RawArray(eeg_data, info)
        return raw_data


def get_signals_from_edf(filename: str) -> RawEDF:
    """
    Reads measurements from edf file and map it to DataFrame

    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawEDF with signals
    """
    edf_data = read_raw_edf(filename)
    # drop all channels which aren't common for all dataset
    edf_data.drop_channels(set(edf_data.ch_names) - set(common_channels))
    return edf_data


def filter_mne(signals: mne.io.RawArray | RawEDF, cutoff_freq: int = 64) -> mne.io.RawArray | RawEDF:
    """
    Performs lowpass filter on all signals.

    :param signals: Dataframe with all signals in dataset
    :param cutoff_freq: Frequency from which we want to cutoff frequencies
    :return: Mne object with applied filter, object is the same that is passed
    """
    signals.filter(1, cutoff_freq, picks='eeg')
    return signals


def save_to_pickle_file(signals: mne.io.RawArray | RawEDF, filename: str) -> None:
    dump(dict(zip(signals.ch_names, signals.get_data())), filename)


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
    eeg_data_list = signals.get_data()
    channel_names = signals.ch_names

    placeholder_data = []
    for eeg_data in eeg_data_list:
        placeholder_data.append(resample_list(eeg_data))

    channel_types = ['eeg'] * len(channel_names)

    info = mne.create_info(ch_types=channel_types, sfreq=new_sample_rate, ch_names=channel_names)
    raw_data = mne.io.RawArray(placeholder_data, info)
    return raw_data


def scale_values(signals: mne.io.RawArray | RawEDF, min_val: float = 0, max_val: float = 1) -> mne.io.RawArray:
    data = signals.get_data()
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_normalized = data_normalized * (max_val - min_val) + min_val

    info = mne.create_info(ch_types=signals.get_channel_types(), sfreq=signals.info['sfreq'], ch_names=signals.ch_names)
    raw_data = mne.io.RawArray(data_normalized, info)
    return raw_data


def split_into_time_windows(signal: mne.io.RawArray, sample_frequency: int, secs: float = 3) -> List[float]:
    """
    Returns some time window from signal. It's generator for simplicity’s sake.

    :param signal: Values from one channel.
    :param sample_frequency: Frequency at which data is sampled.
    :param secs: By what time divide data
    :return: Signal from time chunk
    """

    time = len(signal) / sample_frequency

    chunks = int(time // 3)

    data = signal.get_data()

    for each_channel in data:
        for i in range(chunks - 1):
            # [chunk_start : chunk_end]
            yield each_channel[(secs * sample_frequency) * i: (secs * sample_frequency) * (i + 1)]


def process_folder(folder_name: str, mode: Literal['edf', 'csv', 'eea']) -> None:
    file_names = os.listdir(folder_name)

    if mode == 'edf':
        for file_name in file_names:
            data = get_signals_from_edf(os.path.join(folder_name, file_name))

    elif mode == 'csv':
        for file_name in file_names:
            data = get_signals_from_csv(os.path.join(folder_name, file_name))
            for signal_chunk in data:
                pass
    elif mode == 'eea':
        for file_name in file_names:
            data = get_signals_from_eea(os.path.join(folder_name, file_name))


def preprocess_data_all_steps(signals: mne.io.RawArray | RawEDF, path_to_save_data: str):
    # TODO: Add reference electrode
    data = filter_mne(signals)
    data = resample_signal(data, original_sample_rate=signals.info['sfreq'])
    data = scale_values(data)
    for time_window in split_into_time_windows(data, sample_frequency=signals.info['sfreq']):
        save_to_pickle_file(time_window, path_to_save_data)


def calculate_margenau_lib(signal: List[float]) -> List[List[float]]:
    """
    Calculate MH distribution using TFTB package

    :param signal:
    :return:
    """
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    return tfr_real.tfr
