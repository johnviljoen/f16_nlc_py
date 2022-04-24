import torch
import os

NN = {}

print('loading neural network lookup tables...')

for file in os.listdir('tables/NNs'):
    if file.endswith('.pt') and file != 'lookup_table_limits.pt':
        file_no_ext = os.path.splitext(file)[0]
        NN[file_no_ext] = torch.load(f'tables/NNs/{file}')['model']

print('SUCCESS: neural nets loaded')
