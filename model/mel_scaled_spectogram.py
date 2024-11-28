import librosa
from librosa import feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mel_scaled_spectrogram(eeg_signal :np.ndarray, sampling_rate:int):
    mel_spectrogram = librosa.feature.melspectrogram(y=eeg_signal, sr=sampling_rate, n_mels=64, n_fft=256, hop_length=64)
    return mel_spectrogram

def print_mel_scaled_sepectrogram(data: np.ndarray, sampling_rate: int):
    fig, ax = plt.subplots()
    s_dB = librosa.power_to_db(data, ref=np.max)
    img = librosa.display.specshow(s_dB, x_axis='time',
                                   y_axis='mel', sr=sampling_rate, n_fft=256, hop_length=64,
                                   fmax=sampling_rate / 2, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
