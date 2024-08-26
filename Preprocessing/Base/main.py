from converter import *
from plotter import plot_signal

# signals = get_signals_from_edf("h01.edf")

# print(signals)

result = math.log(250, 2)

result_int = int(result)

# if numbers are not the same it means that in log function is reminder
# which means that number is not the power of two
print(result == result_int)
