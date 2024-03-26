import numpy as np 
import pandas as pd
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

# create Dataset class from df, where Time and Flux are the timeseries data and Spectral Type is the label

def encode_labels(labels):
    # Create a mapping from labels to integers
    unique_labels = sorted(set(labels))  # Sort to ensure consistency
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx

class LightCurveDataset(Dataset):
    def __init__(self, dataframe, label_to_idx):
        self.dataframe = dataframe
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        time = self.dataframe.iloc[idx]['Time']
        flux = self.dataframe.iloc[idx]['Flux']
        time_series = torch.tensor([time, flux], dtype=torch.float)

        spectral_type = self.dataframe.iloc[idx]['Spectral Type']
        label = self.label_to_idx[spectral_type]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'time_series': time_series, 'label': label}

        return sample

if __name__ == '__main__':
    df = pd.read_hdf('tessOstars.h5', 'df')
    label_to_idx = encode_labels(df['Spectral Type'])
    dataset = LightCurveDataset(df, label_to_idx)
    # Example of utilizing __getitem__
    sample = dataset[0]  # This uses the idx to fetch the first sample
    print(sample)
