import torch
import torch.nn as nn
import numpy as np 
import wandb
from run_mlp import load_data, preprocess_data, createdataloaders
import self_supervised
from torch.optim.lr_scheduler import ReduceLROnPlateau
from run_cnn import test_loop, train_loop

torch.manual_seed(42)
np.random.seed(42)

class CNN1DFrozenConv(nn.Module):
    def __init__(self, pretrained_encoder, output_size, input_size, device):
        super().__init__()
        self.encoder = pretrained_encoder
        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
         # use a dummy input to dynamically determine the output dimension
        dummy_input = torch.randn(1, 1, input_size, device=device)  # batch size of 1, 1 channel, and initial input size
        dummy_output = self.encoder(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]  # total number of features divided by batch size
        # add classification layers
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),  
            # nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.encoder(x) 
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-self-supervised", entity="rczhang")
    # data preprocessing steps
    power, logpower, labels, freq = load_data('tessOBAstars.h5')
    power_tensor, labels_tensor, label_to_int = preprocess_data(power, logpower, labels, freq)
    batch_size = 64
    train_dataloader, test_dataloader, class_weights = createdataloaders(power_tensor, labels_tensor, batch_size, augment_data=False, additive=False)
    # create model
    num_channels = 32
    input_size = len(power.iloc[0]) 
    print('input size in cnn_selfsup', input_size)
    # encoder_output_dim = num_channels * 2 * (input_size // 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### manually modify size ###
    pretrained_model = self_supervised.SimCLR(self_supervised.EncoderCNN1D(num_channels, input_size), 256)
    pretrained_model.load_state_dict(torch.load('best_selfsup42_2conv.pth', map_location=device))
    pretrained_model.to(device)
    model = CNN1DFrozenConv(pretrained_model.encoder, len(label_to_int), input_size, device).to(device)
    # loss function 
    best_loss = float('inf')
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    t = -1
    current_loss = test_loop(test_dataloader, model, loss_fn, t, label_to_int)
    # optimizer only updates parameters of non-convolutional layers
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.fc_layers.parameters(), lr=learning_rate)
    patience = 200 # number of epochs to wait for improvement before stopping
    patience_counter = 0
    epochs = 500
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        current_loss = test_loop(test_dataloader, model, loss_fn, t, label_to_int)
        scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_cnn_selfsup.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    print("Done!")
    wandb.finish()