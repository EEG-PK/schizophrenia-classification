import os
from typing import List, Literal
import matplotlib.pyplot as plt


def plot_signal(signal: List) -> None:
    plt.imshow(signal, aspect='auto', cmap='viridis', origin='lower')
    plt.axis((0, len(signal[0]), 0, 128))
    plt.show()


def save_signal_to_file(folder: Literal['PlotsMne', 'PlotsManualDownsampling'], batch_number: int, filename: str,
                        signal: List) -> None:
    plt.imshow(signal, aspect='auto', cmap='viridis', origin='lower')
    plt.axis((0, len(signal[0]), 0, 128))
    if not os.path.exists(f"{folder}/Batch {batch_number}"):
        os.mkdir(f"{folder}/Batch {batch_number}")
    plt.savefig(f"{folder}/Batch {batch_number}/{filename}")
