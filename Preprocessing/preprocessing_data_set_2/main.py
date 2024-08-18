from preprocessing_method import *

prefixes = ['h', 's']
numbers = [f'{i:02d}' for i in range(1, 15)]
file_list = [f"{prefix}{num}.edf" for prefix in prefixes for num in numbers]

for file_name in file_list:
    folder_path = 'dataset/' + file_name
    print(folder_path)
    preprocessing_data_set_2(folder_path)