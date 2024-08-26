from converter import *
from plotter import plot_signal
from joblib import dump, load

signals = get_signals_from_csv('CsvData/25 trimmed.csv')

for signal in signals:
   process_folder('CsvData','csv')