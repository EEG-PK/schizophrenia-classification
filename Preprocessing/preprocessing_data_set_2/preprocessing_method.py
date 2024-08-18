import mne as mne
import numpy as np
from scipy import signal
from scipy.signal import stft
from scipy.fft import fft, fftfreq
import tftb
from tftb.processing import MargenauHillDistribution
from tftb.generators import atoms
import matplotlib.pyplot as plt

def preprocessing_data_set_2(filePath):
    raw = mne.io.read_raw_edf(filePath, preload=True)
    raw.set_montage('standard_1020')
    raw.filter(0.5, 40, picks='eeg')
    raw.resample(sfreq=128)  # downsampling
    channels_names = ['F8', 'O2', 'F7', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Cz', 'Pz'] # channels which are the same for 3 data sets
    data = raw.get_data(picks=channels_names)

    filename = 'preprocessed_data/' + filePath[8:11] + '.csv'
    np.savetxt(filename, data, delimiter=',', fmt='%.8e')


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


def margenau_hill_distribution_image(signal, extend=True):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

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


def margenau_hill_distribution_log_image(signal, extend=True):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    # logarithmizing the results
    epsilon = 1e-10  # Small value to avoid log(0) problems
    tfr_real.tfr = np.log(tfr_real.tfr + epsilon)
    threshold = -20

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


def margenau_hill_distribution_log_normalized_image(signal, extend=True):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()

    # Normalize the results by dividing by the maximum value
    tfr_max = np.amax(tfr_real.tfr)
    tfr_real.tfr = tfr_real.tfr / tfr_max if tfr_max != 0 else tfr_real.tfr
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
