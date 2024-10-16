# schizophrenia-classification
Classification of schizophrenia by EEG signals using CNN network

## Introduction
This study aims to classify schizophrenia using convolutional neural networks (CNN). One method of detecting schizophrenia is the use of electroencephalography (EEG), which records the brain's electrical activity and provides information about differences in brain function between affected and healthy individuals. EEG data will be transformed into the time-frequency domain using the Margenau-Hill distribution (MH-TFD), which allows for the analysis of nonlinear and non-stationary signals. The resulting two-dimensional array data will be fed into a CNN model capable of autonomously detecting brain activity features. It is also worth mentioning that deep neural networks are also resistant to computational errors during feature extraction compared to traditional learning methods. The convolutional network will be trained and tested on three publicly available datasets. The final solution will be applicable to data from other studies with different characteristics, thanks to methods that minimize the influence of the reference electrode selection.

## Datasets
### Dataset "number 1":
- https://www.kaggle.com/broach/buttontonesz2
- EDF channel names: ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'VEOa', 'VEOb', 'HEOL', 'HEOR', 'Nose', 'TP10']
- Number of channels: 64 (ONLY for subject 21, the rest have less).
- Number of patients: 49
- Number of schizophrenia instances: 36
- Number of instances of normal: 13

### Dataset "number 2":
- https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441
- EDF channel names: ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']
- Number of channels: 19
- Number of patients: 28
- Number of schizophrenia instances: 14
- Number of instances of normal: 14
- Number of samples: 231250
- Description from page: The dataset comprised 14 patients with paranoid schizophrenia and 14 healthy controls. Data were acquired with the sampling frequency of 250 Hz using the standard 10-20 EEG montage with 19 EEG channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2. The reference electrode was placed between electrodes Fz and Cz.

### Dataset "number 3":
- http://brain.bio.msu.ru/eeg_schizophrenia.htm
- Channel names: [F7, F3, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2]
- Number of channels: 16
- Number of patients: 84
- Number of schizophrenia instances: 45
- Number of instances of normal: 39
- Number of samples: 7680
- Description from page: Each file contains an EEG record for one subject. Each TXT file contains a column with EEG samples from 16 EEG channels (electrode positions). Each number in the column is an EEG amplitude (mkV) at distinct sample. First 7680 samples represent 1st channel, then 7680 - 2nd channel, ets. The sampling rate is 128 Hz, thus  7680 samples refer to 1 minute of EEG record.

## Code
The model was trained on Debian 12 on the GPU.
The code was tested on Python3.11 and CUDA 12.2.<br>
CUDA and cudnn are installed in the conda environment. All you need is NVIDIA drivers compatible with at least CUDA 12.2 and the Linux environment.

### Setup
#### 1) Create environment
Conda<br>
`conda env create --file ml_eeg_gpu_tf215_ver.yml`<br>
`conda activate ml_eeg_gpu_tf215_ver`
#### 2) Set the `LD_LIBRARY_PATH` environment variable
```bash
chmod +x env_create.sh
./env_create.sh
conda deactivate
conda activate ml_eeg_gpu_tf215_ver
```
#### 3) Fix tftb library compatibility
`python tftb_repair.py`

## Tasks
### Task 1: EEG signal preparation/preprocessing<a id='task-1'></a>
1. Go to the `preprocessing` folder.<br>
`cd preprocessing`
2. Download all datasets using command:<br>
`python dataset_setup.py`<br>
This command will automatically download and unzip all datasets into appropriate folders<br><br>

3. After that run<br>
`python main.py -m {mode} -ps {patient_state} -i {folder_path}`<br>
or move all files to one folder and run<br>
`python main.py -m all -ps {patient_state} -i {folder_path}`<br><br>
For example, for 'number 3' (_.eea_) dataset:<br>
`python main.py -m Eea -ps health -i Datasets/EeaHealthy`<br>
`python main.py -m Eea -ps ill -i Datasets/EeaIll`<br><br>
By default all file should be contained folder called Data in root of preprocessing folder.<br>
What is important that in one folder there can only one state of patients (either healthy or ill). 

### Task 2: Schizophrenia prediction
1. Prepare data according to [preprocessing](#task-1) step.<br>
2. Go to the `model` folder.<br>
`cd model`
3. Search for optimal hyperparameters<br>
`python main.py`
4. Check out the results<br>
`optuna-dashboard sqlite:///schizo_model.db`<br>
Open Optuna's dashboard:<br>
`http://localhost:8080`

[//]: # (#### Training)

[//]: # (`python train.py`)

[//]: # (#### Testing)

[//]: # (`python test.py`)
