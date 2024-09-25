import argparse

parser = argparse.ArgumentParser(description='Preprocess datasets for specific formats')

parser.add_argument('-m', '--mode', help='Mode of preprocessing which app should be run', choices=['Csv', 'Eea', 'Edf', 'All'], required=True)
parser.add_argument('-i', '--input', help='Path to input dataset', default='Data', type=str)
parser.add_argument('-o', '--output', help='Path to out preprocessed dataset', default=None, type=str)
parser.add_argument('-ps', '--patient-state', help='State of patients which are about to be preprocessed', required=True, choices=['health', 'ill'])