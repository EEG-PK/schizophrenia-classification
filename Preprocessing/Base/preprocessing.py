import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal


def read_eeg_data(file_path):
    df = pd.read_csv(file_path, header=None, sep='\s+')
    num_samples_per_channel = 7680
    num_channels = 16

    # Podział danych na poszczególne kanały
    channels = []
    for i in range(num_channels):
        start_index = i * num_samples_per_channel
        end_index = (i + 1) * num_samples_per_channel
        channel_data = df.iloc[start_index:end_index, 0].values
        channels.append(channel_data)

    filtered_channels = filter_eeg_signal(channels)
    return filtered_channels


def filter_eeg_signal(channels):
    # Zastosowanie filtracji pasmowoprzepustowej (np. 0.5 - 40 Hz)
    filtered_channels = []
    fs = 128  # Częstotliwość próbkowania (Hz)
    lowcut = 0.5  # Dolna granica filtru (Hz)
    highcut = 40.0  # Górna granica filtru (Hz)

    for channel_data in channels:
        filtered_data = butter_bandpass_filter(channel_data, lowcut, highcut, fs)
        filtered_channels.append(filtered_data)

    # Zwrócenie listy zawierającej przefiltrowane dane dla każdego kanału
    return filtered_channels


# Jakaś przykładowa filtracja
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def margenau_hill_distribution(signal):
    analytic_signal = hilbert(signal)
    z = np.outer(analytic_signal, np.conjugate(analytic_signal))
    return np.abs(z)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def signal_to_image(signal, flip=False):
    mh_distribution = margenau_hill_distribution(signal)
    img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)  # odwrócenie częstotliwości
    img = 255 - img  # odwrócenie kolorów
    return img


def split_image(image, segment_size):
    segments = []
    for start in range(0, image.shape[1] - segment_size + 1, segment_size):
        segment = image[:, start:start + segment_size]
        segments.append(segment)
    return np.array(segments)
