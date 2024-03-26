import numpy as np 
import pandas as pd
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
# from read_tess_data import 

# convert raw data of two lists into usable data for training
def create_data(raw_data):
    data = []
    for i in range(len(raw_data[0])):
        data.append([raw_data[0][i], raw_data[1][i]])
    return data

# create a dataset class for the TESS data


class TESSDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = np.load(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
