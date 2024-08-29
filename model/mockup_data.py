import joblib
import pandas as pd

from params import SCHIZO_DUMP_FILE, HEALTH_DUMP_FILE


def read_eeg_data(file_path):
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    num_samples_per_channel = 7680
    num_channels = 16

    # Podział danych na poszczególne kanały
    channels = []
    for i in range(num_channels):
        start_index = i * num_samples_per_channel
        end_index = (i + 1) * num_samples_per_channel
        channel_data = df.iloc[start_index:end_index, 0].values
        channels.append(channel_data)

    return channels


def mockup(data_dir, ill_files, health_files, dataset_dir):
    eeg_data_schizophrenia = [{'id': fine_name, 'eeg': read_eeg_data(f'{data_dir}/{fine_name}')} for fine_name in
                              ill_files]
    eeg_data_health = [{'id': fine_name, 'eeg': read_eeg_data(f'{data_dir}/{fine_name}')} for fine_name in health_files]

    f = open(f"{dataset_dir}/{SCHIZO_DUMP_FILE}", "wb")
    joblib.dump(eeg_data_schizophrenia, f)
    f.close()
    f = open(f"{dataset_dir}/{HEALTH_DUMP_FILE}", "wb")
    joblib.dump(eeg_data_health, f)
    f.close()
