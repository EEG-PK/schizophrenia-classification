from converter import *
from plotter import plot_signal

signals = get_signals_from_edf("h01.edf")
filtered = lowpass_filter(signals, 250)
resampled = resample_signal(filtered, 250)
split_into_time_windows(resampled['Fp1'], 128)

gen = split_into_time_windows(resampled['Fp1'], 128)
values = next(gen)
margenau = calculate_margenau_lib(values)
plot_signal(margenau)
