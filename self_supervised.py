from torch import nn
import torch
from lightly.models.modules import SimCLRProjectionHead
import h5py
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from run_mlp import moving_average
import wandb

torch.manual_seed(42)
np.random.seed(42)

def read_hdf5_data(hdf5_path):
    power = []
    spectral_types = []
    with h5py.File(hdf5_path, 'r') as h5f:
        if 'Frequency' in h5f:
            frequencies = np.array(h5f['Frequency'])
        for name in h5f:
            if name != 'Frequency': 
                dataset = h5f[name]
                power.append(np.array(dataset))
                spectral_types.append(dataset.attrs['Spectral_Type'])
    return power, spectral_types, frequencies

def add_noise(data, noise_level=0.01):
    noise = 1 + np.random.uniform(-noise_level, noise_level, data.shape)
    return data * noise

class PowerDataset(Dataset):
    def __init__(self, power, transform=None):
        self.power = power
        self.transform = transform
    
    def __len__(self):
        return len(self.power)

    def __getitem__(self, idx):
        x = self.power[idx]
        if self.transform:
            x_i = self.transform(np.array(x))
            x_j = self.transform(np.array(x))
        else:
            x_i, x_j = x, x
        x_i = np.expand_dims(x_i, axis=0)
        x_j = np.expand_dims(x_j, axis=0)
        return torch.tensor(x_i, dtype=torch.float32), torch.tensor(x_j, dtype=torch.float32)

class EncoderCNN1D(nn.Module):
    def __init__(self, num_channels, input_size):
        super(EncoderCNN1D, self).__init__()
        print('input size', input_size)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, num_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.MaxPool1d(kernel_size=2), # 06/07: TESTING ONE MORE POOLING LAYER
        )
        # use a dummy input to dynamically determine the output dimension
        dummy_input = torch.randn(1, 1, input_size)  # batch size of 1, 1 channel, and initial input size
        dummy_output = self.conv_layers(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]  # total number of features divided by batch size
        print('dummy output shape', dummy_output.shape)
        print('output dim', self.output_dim)
        # self.output_dim = num_channels * 2 * (input_size // 4)

    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
class SimCLR(nn.Module):
    def __init__(self, encoder, embedding_dim=256):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i.view(h_i.size(0), -1))
        z_j = self.projector(h_j.view(h_j.size(0), -1))

        return h_i, h_j, z_i, z_j


def contrastive_loss(z_i, z_j, temperature=0.1):
    cos = nn.CosineSimilarity(dim=-1)
    sim = cos(z_i.unsqueeze(1), z_j.unsqueeze(0)) / temperature
    sim_i_j = torch.diag(sim)
    loss = -torch.log(torch.exp(sim_i_j) / torch.exp(sim).sum(dim=1))
    return loss.mean()

def train_model(model, dataloader, optimizer, epochs):
    best_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j) in dataloader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = contrastive_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"Epoch": epoch+1, "batch_loss": loss.item()})
        avg_loss = total_loss / len(dataloader)
        wandb.log({"Epoch": epoch+1, "Avg Loss": avg_loss})
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "best_selfsupervised.pth")
        print(f'Epoch {epoch+1}, Avg Loss: {avg_loss}')

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-self-supervised", entity="rczhang")
    power, labels, freq = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOBAstars_all.h5')
    smoothed_power = moving_average(power, 10)
    print('length of stars', len(smoothed_power))
    # apply stochastic data augmentations to each list of powers, multiplicative noise uniformly distributed between 0.99 and 1.01x the data
    power_dataset = PowerDataset(smoothed_power, transform=add_noise)
    dataloader = DataLoader(power_dataset, batch_size=500, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = 32
    input_size = len(power[0])
    encoder = EncoderCNN1D(num_channels=num_channels, input_size=input_size)
    model = SimCLR(encoder=encoder, embedding_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, dataloader, optimizer, 50)
    wandb.finish()