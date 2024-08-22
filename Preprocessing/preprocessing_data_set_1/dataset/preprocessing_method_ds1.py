import os
import numpy as np
import mne
from scipy.signal import stft, spectrogram
import matplotlib.pyplot as plt


def read_eea(file_name, samples=7680, channels=16):
    data = []
    with open(file_name, 'r') as eea:
        lines = eea.readlines()
        for c in range(channels):
            channel = []
            for s in range(samples):
                channel.append(float(lines[c * samples + s].strip()))
            data.append(channel)
    data = np.array(data)
    return np.transpose(data)


def preprocessing_data_set_eeg(data, sfreq=128, channels_names=None):
    if channels_names is None:
        channels_names = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

    info = mne.create_info(channels_names, sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)

    raw.filter(0.5, 40, picks='eeg')
    raw.resample(sfreq=sfreq)

    data = raw.get_data(picks=channels_names)
    return data


def plot_stft(signal, sfreq=128, nperseg=256, noverlap=128, output_path=None):
    f, t, Zxx = stft(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 60)  # 60 Hz
    plt.colorbar(label='Magnitude')

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def process_all_files_in_folder(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith('.eea'):
            file_path = os.path.join(input_folder_path, filename)
            eeg_data = read_eea(file_path)

            processed_data = preprocessing_data_set_eeg(eeg_data)

            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}.png")
            plot_stft(processed_data[0, :], output_path=output_file_path)
            print(f"Zapisano obraz do: {output_file_path}")


if __name__ == "__main__":
    input_folder_path = 'C:\\Users\\huber\\Downloads\\schizophrenia-classification-main\\schizophrenia-classification-main\\Preprocessing\\preprocessing_data_set_1\\dataset\\norm'
    output_folder_path = 'C:\\Users\\huber\\Downloads\\schizophrenia-classification-main\\schizophrenia-classification-main\\Preprocessing\\preprocessing_data_set_1\\dataset\\norm_after_processing'
    process_all_files_in_folder(input_folder_path, output_folder_path)
