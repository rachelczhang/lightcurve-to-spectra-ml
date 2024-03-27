import numpy as np 
import pandas as pd
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
        time_flux_stack = np.stack((time, flux), axis=0)  # This creates a single numpy array with shape (2, N)
        time_flux = torch.tensor(time_flux_stack, dtype=torch.float)
        # time_flux = torch.tensor([time, flux], dtype=torch.float)

        spectral_type = self.dataframe.iloc[idx]['Spectral Type']
        label = self.label_to_idx[spectral_type]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'time_flux': time_flux, 'spectral_type_label': label}

        return sample
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    df = pd.read_hdf('tessOstars.h5', 'df')
    # separate data into training and testing datasets
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    print('train df', train_df)
    print('test df', test_df)
    train_label_to_idx = encode_labels(train_df['Spectral Type'])
    print('train label', train_label_to_idx)
    test_label_to_idx = encode_labels(test_df['Spectral Type'])
    train_dataset = LightCurveDataset(train_df, train_label_to_idx)
    test_dataset = LightCurveDataset(test_df, test_label_to_idx)

    # set hyperparameters: adjustable parameters to control model optimization process
    # number of epochs: # times to iterate over datatset
    # batch size: # data samples propagated through network before parameters updated
    # learning rate: how much to update models parameters at each batch/epoch
    learning_rate = 0.001
    batch_size = 64
    epochs = 5
    input_size = len(train_dataset[0]['time_flux'].flatten())  # Adjust based on your data
    output_size = len(train_label_to_idx)
    print('input size', input_size)
    print('output size', output_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MLP(input_size, output_size)
    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()
    # optimization: process of adjusting model parameters to reduce model error in each training step
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # # training
    # for epoch in range(epochs):
    #     for i, batch in enumerate(train_loader):
    #         # flatten data to fit the MLP input
    #         print('og', batch['time_flux'])
    #         inputs = batch['time_flux'].view(batch_size, -1)
    #         print('inputs', inputs)
    #         labels = batch['spectral_type_label']
    #         print('labels', labels)
            
    #         # Forward pass
    #         outputs = model(inputs)
    #         loss = loss_fn(outputs, labels)
            
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         break        
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    #     break 
