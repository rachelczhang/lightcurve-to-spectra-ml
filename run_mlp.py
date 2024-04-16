import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

def load_data(filename):
    """
    Directly read the data saved from read_tess_data.py
    """
    data = pd.read_hdf(filename, 'df')
    power = data['Power']
    print('data', data)
    freq = data['Frequency']
    labels = data['Spectral Type']
    return power, labels, freq


def encode_labels(labels):
    """
    Encode string labels into unique integers using PyTorch.
    
    Parameters:
    labels: DataFrame series of string labels.
    
    Returns:
    torch.Tensor: encoded labels as a tensor of long integers.
    dict: a mapping from original labels to encoded integers.
    """
    unique_labels = labels.unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels_encoded = labels.map(label_to_int).values
    return torch.tensor(labels_encoded, dtype=torch.long), label_to_int

def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate weights for each class based on frequencies.
    
    Parameters:
    labels: torch.Tensor - Encoded labels as a tensor of long integers.
    
    Returns:
    torch.Tensor: Weights for each class.
    """
    class_counts = labels.bincount()
    total_samples = len(labels)
    num_classes = len(class_counts)
    weights = total_samples / (class_counts * num_classes)
    return weights

def moving_average(data, window_size):
    """
    Apply moving average smoothing to the data.
    
    Parameters:
    data: data to be smoothed: list of lists of floats
    window_size: int - Size of the moving average window.
    
    Returns:
    np.array: Smoothed data.
    """
    window = np.ones(window_size) / window_size
    smoothed_data = []
    for sublist in data:
        smoothed_sublist = np.convolve(sublist, window, mode='same')
        smoothed_data.append(list(smoothed_sublist))
    return smoothed_data

def plot_spectra(og_power, conv_power, freq):
    """
    Plot the original and smoothed spectra.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(freq[0], og_power[0], label='Original Data', alpha=0.5)
    plt.plot(freq[0], conv_power[0], label='Moving Average', alpha=0.8)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [1/d]')
    plt.ylabel('Amplitude Power')
    plt.savefig('spectratest.png')
    plt.clf()

def preprocess_data(power, labels, freq):
    """
    Convert power data to tensor usable for PyTorch and encode labels, then make them DataLoaders 
    to be directly used in training and testing

    Parameters:
    power: DataFrame series of lists of floats 
    labels: DataFrame series of strings 

    Returns: 
    torch.Tensor: Tensor of power data
    """
    # convert DataFrame series of lists of floats to tensor
    og_power_array = np.array(power.tolist()) 
    print('og len', len(og_power_array))
    power_array = moving_average(og_power_array, 10)
    print('ma len', len(power_array))
    # plot_spectra(og_power_array, power_array, freq)
    power_tensor = torch.tensor(power_array, dtype=torch.float32)
    
    # encode string labels to integers
    labels_tensor, label_to_int = encode_labels(labels)
    class_weights = calculate_class_weights(labels_tensor)

    # combined Dataset
    dataset = TensorDataset(power_tensor, labels_tensor)

    # split data into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_to_int, class_weights

# define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            # nn.Softmax()
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Runs one epoch of training: goes through entire dataset once, updating model weights 
    based on loss calculated from predictions
    
    Parameters:
    dataloader: iterates over dataset, providing batches of data (X) and labels (y)
    model: NN model that is being trained
    loss_fn: loss function used to evaluate models' predictions against true labels
    optimizer: optimization algorithm used to update model weights based on computed gradients
    """
    # number of samples in dataset
    size = len(dataloader.dataset)
    # puts model in training mode, which is important for dropout layers and batch normalization layers
    model.train()
    # iterates over each batch in dataloader; X is a batch of input data, y is the corresponding batch of target labels
    for batch, (X, y) in enumerate(dataloader):
        # get predictions and loss
        X = X.cuda()
        y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)
        # backpropagation: compute gradient of loss with respect to each weight in model
        loss.backward()
        # updates model weights based on gradients calculated in loss.backward()
        optimizer.step()
        # clears gradients to prevent accumulation (?? don't understand)
        optimizer.zero_grad()
        # prints current loss and progress every 100 batches; loss.item() converts loss from tensor to number
        # current = number of samples processed so far
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    """
    Evaluate the model's performance on the test dataset
    """
    # put model in evaluation mode
    model.eval()
    # number of samples in dataset
    size = len(dataloader.dataset)
    # number of batches in dataloader
    num_batches = len(dataloader)
    # iniitalizes variables to accumulate total loss and number of correctly predicted samples
    test_loss, correct = 0, 0

    # torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        # iterate over each batch in dataloader
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()
            # calculate loss, add to test_loss, compute # correct predictions
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # normalizes test_loss by number of batches and calcules accuracy as percentage of correct predictions
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    power, labels, freq = load_data('tessOBAstars.h5')
    learning_rate = 1e-1
    batch_size = 64
    epochs = 20
    train_dataloader, test_dataloader, label_to_int, class_weights = preprocess_data(power, labels, freq)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()#, reduction='sum')
    model = MLP(input_size=len(power.iloc[0]), output_size=len(label_to_int)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
