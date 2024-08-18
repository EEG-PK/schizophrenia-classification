import numpy as np
from scipy.signal import hilbert, get_window
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from tftb.processing import MargenauHillDistribution
from tftb.generators import atoms
from sklearn.decomposition import PCA

from preprocessing import filter_eeg_signal

SIGNAL_LEN = 7680
NUM_CHANNELS = 16
SAMPLING_RATE = 128


# Version 5 (Discord - Hubert)
def margenau_hill_distribution(signal, window='hann', nfft=1024, scaling_factor=10):
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
    # Normalisation
    # tfr = tfr / np.max(tfr)

    plt.imshow(tfr.T, aspect='auto', cmap='viridis', origin='lower')
    plt.show()
    image = convert_to_image(tfr.T, flip=True)
    plt.imshow(image, aspect='auto', cmap='gray')
    plt.show()

    return tfr


# Version 4 (tftb library)
def margenau_hill_distribution_image(signal, extend=True):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()
    # tfr_real.plot(show_tf=False, kind='cmap', sqmod=False, threshold=0)

    threshold = 0.05
    tfr_real.tfr = tfr_real.tfr[:(tfr_real.tfr.shape[0] // 2), :]
    _threshold = np.amax(tfr_real.tfr) * threshold
    tfr_real.tfr[tfr_real.tfr <= _threshold] = 0.0
    extent = (0, tfr_real.ts.max(), 0, 0.5)
    plt.imshow(tfr_real.tfr, aspect='auto', cmap='viridis', origin='lower', extent=extent)
    plt.show()
    image = convert_to_image(tfr_real.tfr, flip=False)
    plt.imshow(image, aspect='auto', cmap='gray', origin='lower', extent=extent)
    plt.show()

    if extend:
        return image, extent
    else:
        return image


# Version 1
# def margenau_hill_distribution(signal):
#     analytic_signal = hilbert(signal)
#     z = np.outer(analytic_signal, np.conjugate(analytic_signal))
#     return np.abs(z)


# Version 2
# def margenau_hill_distribution(signal):
#     # Obliczanie krótkoczasowej transformaty Fouriera (STFT)
#     # f, t, Zxx = stft(signal, nperseg=len(signal)//4)
#     f, t, Zxx = stft(signal, nperseg=256)
#     # Obliczanie rozkładu Margenau-Hill jako kwadratu modułu STFT
#     mh_distribution = np.abs(Zxx) ** 2
#     return mh_distribution

# Version 3
# def margenau_hill_distribution(signal, fs):
#     # Obliczenie transformacji Fouriera sygnału y(t)
#     Y = fft(signal)
#
#     # Długość sygnału
#     N = len(signal)
#
#     # Częstotliwości
#     f = fftfreq(N, d=1/fs)
#
#     # Indeksy częstotliwości dodatnich (0-64 Hz)
#     positive_freq_indices = np.where((f >= 0) & (f <= fs/2))[0]
#     f_positive = f[positive_freq_indices]
#
#     # Generowanie rozkładu Margenau-Hill
#     MH_y = np.zeros((N, len(f_positive)), dtype=np.float64)
#     for i in range(N):
#         for j, idx in enumerate(positive_freq_indices):
#             MH_y[i, j] = np.real(signal[i] * np.exp(-1j * 2 * np.pi * f[idx] * i / fs) * Y[idx])
#
#     return MH_y, f_positive


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def convert_to_image(mh_distribution, flip=True):
    img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)
    img = 255 - img  # invert. make black==more energy
    return img


def read_eeg_data(file_path):
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    num_samples_per_channel = SIGNAL_LEN
    num_channels = NUM_CHANNELS

    # Podział danych na poszczególne kanały
    channels = []
    for i in range(num_channels):
        start_index = i * num_samples_per_channel
        end_index = (i + 1) * num_samples_per_channel
        channel_data = df.iloc[start_index:end_index, 0].values
        channels.append(channel_data)

    filtered_channels = filter_eeg_signal(channels)
    return filtered_channels


# Przykład użycia: wczytanie danych dla pliku subject1.txt
# file_path = 'data/485w1.eea'
# eeg_data = np.array(read_eeg_data(file_path))

# Example usage
shortener_signal_len = 1100
signal_length = shortener_signal_len
sampling_rate = SAMPLING_RATE
#
# margenau_hill_distribution(eeg_data[0, :shortener_signal_len], nfft=50)
# image_ch0, extent = margenau_hill_distribution_image(eeg_data[0, :shortener_signal_len])

# image_ch1, _ = margenau_hill_distribution_image(eeg_data[1, :shortener_signal_len])
# image_ch2, _ = margenau_hill_distribution_image(eeg_data[2, :shortener_signal_len])
# image_ch3, _ = margenau_hill_distribution_image(eeg_data[3, :shortener_signal_len])
# MH_y, f = margenau_hill_distribution(eeg_data[0, :shortener_signal_len], sampling_rate)
# image = convert_to_image(tfr_positive, flip=False)

# plt.imshow(image_ch0, aspect='auto', cmap='gray', origin='lower', extent=extent)
# plt.title("Margenau-Hill Distribution - channel 0")
# plt.show()
# plt.imshow(image_ch1, aspect='auto', cmap='gray', origin='lower', extent=extent)
# plt.title("Margenau-Hill Distribution - channel 1")
# plt.show()
# plt.imshow(image_ch1, aspect='auto', cmap='gray', origin='lower', extent=extent)
# plt.title("Margenau-Hill Distribution - channel 2")
# plt.show()

# assert image_ch0.shape == image_ch1.shape == image_ch2.shape == image_ch3.shape
# image_rgb = np.stack((image_ch0, image_ch1, image_ch2, image_ch3), axis=-1)
#
# plt.imshow(image_rgb, aspect='auto', origin='lower', extent=extent)
# plt.show()


#### 2 Spodob ####
# Generowanie obrazów dla każdego kanału (jak wcześniej)
# images = [margenau_hill_distribution_image(eeg_data[i, :shortener_signal_len], extend=False) for i in range(8)]
#
# # Załóżmy, że obrazy są spłaszczone do wektorów 1D
# flattened_images = [image.flatten() for image in images]
# data = np.stack(flattened_images, axis=-1)
#
# # Redukcja wymiarów do 3 za pomocą PCA
# pca = PCA(n_components=3)
# reduced_data = pca.fit_transform(data)
#
# # Przekształcenie z powrotem do formatu obrazu
# image_rgb = reduced_data.reshape(images[0].shape[0], images[0].shape[1], 3)
#
# # Normalizacja do zakresu [0, 1]
# image_rgb = (image_rgb - np.min(image_rgb)) / (np.max(image_rgb) - np.min(image_rgb))
#
# # Wyświetlenie obrazu RGB
# plt.imshow(image_rgb, aspect='auto', origin='lower')
# plt.title("Obraz RGB po redukcji wymiarów")
# plt.axis('off')
# plt.show()
