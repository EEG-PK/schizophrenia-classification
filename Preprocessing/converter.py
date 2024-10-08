import mne
import pandas as pd
from mne.io import read_raw_edf
from mne.utils import set_log_level
from typing import List, Literal, Generator, Dict, Any

from mne.io.edf.edf import RawEDF
from tftb.processing import MargenauHillDistribution
import numpy as np
from joblib import dump
import os

common_channels = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz']


def get_signals_from_csv(filename: str, sample_frequency: int = 1024) -> Generator[mne.io.RawArray, None, None]:
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
        if df.loc[df['condition'] != 2].size != 0:
            continue
        # get only columns which are in common channels in all datasets
        relevant_data = df.iloc[:, 4:-6]
        eeg_data = relevant_data.values.T
        channel_types = ['eeg'] * len(eeg_data)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=csv_labels[4:-6])
        info.set_montage('standard_1020')
        raw_data = mne.io.RawArray(eeg_data, info)
        yield raw_data


def get_signals_from_eea(filename: str, measurements_per_channel: int = 7680,
                         channels: List | None = None, sample_frequency: int = 128) -> mne.io.RawArray:
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
            channels = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6',
                        'O1', 'O2']
        # we need to iterate through all values and skip those which are not common for all datasets
        eeg_data = [lines[i:i + measurements_per_channel] for enum_i, i in
                    enumerate(range(0, len(lines), measurements_per_channel))]

        channel_types = ['eeg'] * len(channels)

        info = mne.create_info(ch_types=channel_types, sfreq=sample_frequency, ch_names=channels)
        info.set_montage('standard_1020')
        raw_data = mne.io.RawArray(eeg_data, info)
        return raw_data


def get_signals_from_edf(filename: str) -> RawEDF:
    """
    Reads measurements from edf file and map it to DataFrame

    :param filename: Path to file. If file is in the project folder name is sufficient.
    :return: RawEDF with signals
    """
    edf_data = read_raw_edf(filename, preload=True)
    edf_data.set_montage('standard_1020')
    return edf_data


def drop_channels(signals: RawEDF | mne.io.RawArray) -> mne.io.RawArray | RawEDF:
    """
    Drops all channels which are not common for all datasets

    :param signals:
    """
    signal_copy = signals.copy().drop_channels(set(signals.ch_names) - set(common_channels))
    return signal_copy


def create_reference_electrode(signals: mne.io.RawArray | RawEDF) -> mne.io.RawArray | RawEDF:
    signals.del_proj()  # remove our average reference projector first
    sphere = mne.make_sphere_model("auto", "auto", signals.info)
    src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0)
    forward = mne.make_forward_solution(signals.info, trans=None, src=src, bem=sphere, meg=False)
    eeg_data = signals.copy().set_eeg_reference(ref_channels='REST', forward=forward)
    # eeg_data = signals.copy().set_eeg_reference(ref_channels='average')

    return eeg_data


def filter_mne(signals: mne.io.RawArray | RawEDF, cutoff_freq: int = 40) -> mne.io.RawArray | RawEDF:
    """
    Performs lowpass filter on all signals.

    :param signals: Dataframe with all signals in dataset
    :param cutoff_freq: Frequency from which we want to cutoff frequencies
    :return: Mne object with applied filter, object is the same that is passed
    """
    signals.filter(1, cutoff_freq, picks='eeg')
    return signals


def save_to_pickle_file(object_to_dump: Any, filename: str, folder_name: str) -> None:
    """
    Saves all the signals into dictionary format to pickle file.

    :param object_to_dump: Any valid python object
    :param filename: Filename of file
    :param folder_name: Folder in which data should be saved. If folder doesn't exist it will be created.
    """
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    dump(object_to_dump, f"{folder_name}/{filename}")


def resample_signal(signals: mne.io.RawArray | RawEDF, original_sample_rate: int,
                    new_sample_rate: int = 128) -> mne.io.RawArray:
    """
    Resample signal using decimate/resample_poly functions


    :param signals: Mne object with signals
    :param original_sample_rate: Sample rate of the original signal
    :param new_sample_rate: New sample rate of which we want make
    :return: Mne object with resampled signals
    """
    # FIXME: currently using resample from mne bcs time window in manual resampling is longer up 4 times for some reason
    signals.resample(new_sample_rate)
    return signals
    # def resample_list(signal_list: List[float]):
    #     """
    #     Resample signal list using decimate/resample_poly
    #
    #     :param signal_list: Signal from one channel. Param should be a list of floats.
    #     """
    #     should_use_poly = original_sample_rate % new_sample_rate != 0
    #
    #     if should_use_poly:
    #         original_sample_rate_int = int(original_sample_rate)
    #         gcd = np.gcd(original_sample_rate_int, new_sample_rate)
    #
    #         # Upsampling and downsampling factors
    #         up = original_sample_rate // gcd
    #         down = new_sample_rate // gcd
    #
    #         # Apply low-pass FIR filter and perform resampling using resample_poly
    #         return resample_poly(signal_list, up, down)
    #     else:
    #         original_sample_rate_int = int(original_sample_rate)
    #         gcd = np.gcd(original_sample_rate_int, new_sample_rate)
    #         down = new_sample_rate // gcd
    #         # Apply low-pass FIR filter and perform resampling using resample_poly
    #         return decimate(signal_list, down)
    #
    # # function returns data | timestamps, timestamps are omitted (not used I guess)
    # eeg_data_list = signals.get_data()
    # channel_names = signals.ch_names
    #
    # placeholder_data = []
    # for eeg_data in eeg_data_list:
    #     placeholder_data.append(resample_list(eeg_data))
    #
    # channel_types = ['eeg'] * len(channel_names)
    #
    # info = mne.create_info(ch_types=channel_types, sfreq=new_sample_rate, ch_names=channel_names)
    # raw_data = mne.io.RawArray(placeholder_data, info)
    # return raw_data


def scale_values(signals: mne.io.RawArray | RawEDF, min_val: float = -1, max_val: float = 1) -> mne.io.RawArray:
    """
    Scales all values to range [a,b]

    :param signals: Signals in RawEdf or RawArray
    :param min_val: Lower bound of range
    :param max_val: Upper bound of range
    :return:
    """
    data = signals.get_data()
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_normalized = data_normalized * (max_val - min_val) + min_val

    info = mne.create_info(ch_types=signals.get_channel_types(), sfreq=signals.info['sfreq'], ch_names=signals.ch_names)
    info.set_montage('standard_1020')
    raw_data = mne.io.RawArray(data_normalized, info)
    return raw_data


def split_into_time_windows(signals: mne.io.RawArray, sample_frequency: int, secs: float = 3) -> Generator[
    dict[str, np.ndarray[float]], None, None]:
    """
    Returns some time window from signal. It's generator for simplicity’s sake.

    :param signals: RawArray from mne with all signals
    :param sample_frequency: Frequency at which data is sampled.
    :param secs: By what time divide data
    :return: Signal from time chunk
    """

    epochs = mne.make_fixed_length_epochs(signals, duration=secs, preload=True)

    for chunk in epochs:
        # zips chunks into dict
        yield dict(zip(epochs.ch_names, chunk))


def cast_from_mne_to_dict(signals: mne.io.RawArray | RawEDF, file_name: str) -> Dict[str, Any]:
    return {'id': file_name, 'eeg': dict(zip(signals.ch_names, signals.get_data()))}


def progress_bar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    This function is taken from stack overflow:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)

    # Progress Bar Printing Function
    def print_progress_bar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    # Initial Call
    print_progress_bar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        print_progress_bar(i + 1)
    # Print New Line on Complete
    print()


def process_edf(input_folder: str, file_name: str) -> Dict[str, np.ndarray]:
    """
    Process single EDF file

    :param input_folder: Folder where file is located
    :param file_name: File name
    :return: Dictionary of EDF data
    """
    data = get_signals_from_edf(os.path.join(input_folder, file_name))
    processed_data = preprocess_data_all_steps(data, file_name)
    return processed_data


def process_eea(input_folder: str, file_name: str) -> Dict[str, np.ndarray]:
    """
    Process single EEA file

    :param input_folder: Folder where file is located
    :param file_name: File name
    :return: Dictionary of EEA data
    """
    data = get_signals_from_eea(os.path.join(input_folder, file_name))
    processed_data = preprocess_data_all_steps(data, file_name)
    return processed_data


def process_csv(input_folder: str, file_name: str) -> List[Dict[str, np.ndarray]]:
    """
    Process single CSV file

    :param input_folder: Folder where file is located
    :param file_name: File name
    :return: List of dictionaries containing signals.
    """
    tmp_dump = []
    data = get_signals_from_csv(os.path.join(input_folder, file_name))
    for index, signal_chunk in enumerate(data):
        processed_data = preprocess_data_all_steps(signal_chunk, f"chunk_{index + 1}_{file_name}")
        tmp_dump.append(processed_data)
    return tmp_dump


def process_folder(input_folder: str, mode: Literal['Edf', 'Csv', 'Eea', 'All'],
                   patient_state: Literal['health', 'ill'], output_folder: str = None) -> None:
    """
    Main function. Read all files from provided folder and process them.
    If output_folder is not provided, it will use default one.

    :param patient_state: Information about the state of patients processed. Needed for file save.
    :param output_folder: Folder where data should be saved.
    :param input_folder: Folder which stores all the files
    :param mode: File extension which are in the folder. THERE SHOULDN'T BE MORE THAN ONE FILE TYPE IN FOLDER
    """
    file_names = os.listdir(input_folder)
    set_log_level("Critical")
    object_dump = []

    if mode == 'Edf':
        for file_name in progress_bar(file_names, prefix='Progress: '):
            data = process_edf(input_folder, file_name)
            object_dump.append(data)
    elif mode == 'Csv':
        for file_name in progress_bar(file_names, prefix='Progress: '):
            data = process_csv(input_folder, file_name)
            object_dump.extend(data)
    elif mode == 'Eea':
        for file_name in progress_bar(file_names, prefix='Progress: '):
            data = process_eea(input_folder, file_name)
            object_dump.append(data)
    elif mode == 'All':
        for file_name in progress_bar(file_names, prefix='Progress: '):
            extension = file_name.split('.')[1]
            if extension == 'csv':
                data = process_csv(input_folder, file_name)
                object_dump.extend(data)
            elif extension == 'eea':
                data = process_eea(input_folder, file_name)
                object_dump.append(data)
            elif extension == 'edf':
                data = process_edf(input_folder, file_name)
                object_dump.append(data)

    if output_folder is not None:
        save_to_pickle_file(object_dump, f'eeg_{mode}_{patient_state}.pk', output_folder)
    else:
        save_to_pickle_file(object_dump, f'eeg_{mode}_{patient_state}.pk', f'{mode}Data')


def preprocess_data_all_steps(signals: mne.io.RawArray | RawEDF, file_name: str) -> Dict[
    str, np.ndarray]:
    """
    Main pipeline for preprocessing

    :param file_name: Name of file which is processed
    :param signals: Signals from files in RawArray or RawEDF
    """
    data = create_reference_electrode(signals)
    # drops all channels which are not common across all datasets
    data = drop_channels(data)
    data = filter_mne(data)
    data = resample_signal(data, original_sample_rate=signals.info['sfreq'])
    data = scale_values(data)
    return cast_from_mne_to_dict(data, file_name)


def calculate_margenau_lib(signal: List[float]) -> List[List[float]]:
    """
    Calculate MH distribution using TFTB package

    :param signal:
    :return:
    """
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    return tfr_real.tfr
