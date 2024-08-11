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
from mh_distribution_image import convert_to_image
from mh_distribution_image import SAMPLING_RATE


def read_csv_file_using_pandas(filename: str, column_name: str) -> np.ndarray:
    data = pd.read_csv(filename)[column_name].to_numpy()
    return data


# z Discorda
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
    # tfr = tfr / np.max(tfr)

    plt.imshow(tfr.T, aspect='auto', cmap='viridis', origin='lower')
    plt.show()
    image = convert_to_image(tfr.T, flip=True)
    plt.imshow(image, aspect='auto', cmap='gray')
    plt.show()

    return tfr


# z tftb
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


shortener_signal_len = 1100
signal_length = shortener_signal_len
sampling_rate = SAMPLING_RATE

eeg_data = read_csv_file_using_pandas('25 trimmed.csv', 'Fp1')
#
margenau_hill_distribution(eeg_data[:shortener_signal_len], nfft=50)
margenau_hill_distribution_image(eeg_data[:shortener_signal_len])
