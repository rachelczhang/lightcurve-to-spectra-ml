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

# create Dataset class from df, where Frequency and Power are the data and Spectral Type is the label

def encode_labels(labels):
    # create a mapping from labels to integers
    unique_labels = sorted(set(labels))  # sort to ensure consistency
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx

class PeriodogramDataset(Dataset):
    def __init__(self, dataframe, label_to_idx):
        self.dataframe = dataframe
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        print('idx', idx)
        freq = self.dataframe.iloc[idx]['Frequency']
        print('freq', freq)
        power = self.dataframe.iloc[idx]['Power']
        print('power', power)
        freq_pow_stack = np.stack((freq, power), axis=0)  # single numpy array with shape (2, N)
        freq_pow = torch.tensor(freq_pow_stack, dtype=torch.float)
        # freq_pow = torch.tensor([time, flux], dtype=torch.float)

        spectral_type = self.dataframe.iloc[idx]['Spectral Type']
        label = self.label_to_idx[spectral_type]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'freq_pow': freq_pow, 'spectral_type_label': label}

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
    test_label_to_idx = encode_labels(test_df['Spectral Type'])
    print('train label to idx', train_label_to_idx)
    print('test label to idx', test_label_to_idx)
    train_dataset = PeriodogramDataset(train_df, train_label_to_idx)
    test_dataset = PeriodogramDataset(test_df, test_label_to_idx)
    print('train dataset', train_dataset)
    print('test dataset', test_dataset)

    # set hyperparameters: adjustable parameters to control model optimization process
    # number of epochs: # times to iterate over datatset
    # batch size: # data samples propagated through network before parameters updated
    # learning rate: how much to update models parameters at each batch/epoch
    learning_rate = 0.001
    batch_size = 64
    epochs = 5
    input_size = len(train_dataset[0]['freq_pow'].flatten())  # Adjust based on your data
    output_size = len(train_label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MLP(input_size, output_size)
    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()
    # optimization: process of adjusting model parameters to reduce model error in each training step
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            # dynamically get current batch size
            current_batch_size = batch['freq_pow'].shape[0]
            # flatten data to fit the MLP input
            inputs = batch['freq_pow'].view(current_batch_size, -1)
            labels = batch['spectral_type_label']
            print('labels', labels)
            
            # forward pass: compute model predictions for batch's inputs
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # backward pass and optimize: compute gradient of loss with respect to model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            print(f'Loss: {loss.item():.4f}')   
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')