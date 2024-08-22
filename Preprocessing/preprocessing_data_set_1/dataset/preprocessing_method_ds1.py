import os
import numpy as np
import mne
from tftb.processing import MargenauHillDistribution
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


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def convert_to_image(mh_distribution, flip=True):
    img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)
    img = 255 - img 
    return img


def margenau_hill_distribution_image(signal, output_path=None):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    threshold = 0.05
    tfr_real.tfr = tfr_real.tfr[:(tfr_real.tfr.shape[0] // 2), :]
    _threshold = np.amax(tfr_real.tfr) * threshold
    tfr_real.tfr[tfr_real.tfr <= _threshold] = 0.0
    extent = (0, tfr_real.ts.max(), 0, 0.5)

    image = convert_to_image(tfr_real.tfr, flip=False)

    plt.imshow(image, aspect='auto', cmap='gray', origin='lower', extent=extent)

    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Zapisano obraz do: {output_path}")
    else:
        plt.show()

    return image


def process_all_files_in_folder(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith('.eea'):
            file_path = os.path.join(input_folder_path, filename)
            eeg_data = read_eea(file_path)
            processed_data = preprocessing_data_set_eeg(eeg_data)
            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}.png")
            margenau_hill_distribution_image(processed_data[0, :], output_path=output_file_path)


if __name__ == "__main__":
    input_folder_path = 'C:\\Users\\huber\\Downloads\\schizophrenia-classification-main\\schizophrenia-classification-main\\Preprocessing\\preprocessing_data_set_1\\dataset\\norm'
    output_folder_path = 'C:\\Users\\huber\\Downloads\\schizophrenia-classification-main\\schizophrenia-classification-main\\Preprocessing\\preprocessing_data_set_1\\dataset\\norm_after_processing'
    process_all_files_in_folder(input_folder_path, output_folder_path)
