from converter import process_folder
from joblib import load
process_folder('Data','Csv', 'health')

# test = load('./CsvData/eeg_Csv_health.pk')
# # print(test)
# print(len(test))
# print(list(test[0].keys()))

