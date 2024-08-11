from typing import List
import matplotlib.pyplot as plt


def plot_signal(signal: List) -> None:
    plt.imshow(signal, aspect='auto', cmap='viridis', origin='lower')
    plt.show()
