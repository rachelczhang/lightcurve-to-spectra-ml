import torch
import torch.nn as nn
from run_mlp import load_data, preprocess_data
import wandb

class CNN1D(nn.Module):
    def __init__(self, num_channels, output_size):
        """
        num_channels: # of output channels for first convolutional layer
        output_size: # of classes for the final output layer
        """
        super().__init__()
        self.conv_layers = nn.Sequential( # container of layers that processes convolutional part of network
            nn.Conv1d(1, num_channels, kernel_size=5, padding=2),  # Convolutional layer with 1 input channel, num_channels output channels, 5 length kernel, and 2 elements of padding on either side
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # max pooling layer that reduces dimensionality by taking max value of each 2-element window
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.flatten = nn.Flatten() # flattens multiple-dimensional tensor convolutional layers output --> 1D tensor
        self.fc_layers = nn.Sequential( # container of layers that processes fully connected part of network
            nn.Linear(num_channels * 2 * (input_size // 4), 128),  
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
    
    def forward(self, x):
        x = self.conv_layers(x) # process inputs through convolutional layers
        x = self.flatten(x) # flattens output of convolutional layers
        logits = self.fc_layers(x) # passes flattened output through fully connected layers to produce logits for each class
        return logits # raw, unnormalized scores for each class

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda().unsqueeze(1)  # format from [batch size, data sample length] ==> [batch size, 1, data sample length], where 1 = one channel per data sample
        y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X) + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f"Train loss: {avg_loss:>7f}")

def test_loop(dataloader, model, loss_fn, epoch):
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
            X = X.cuda().unsqueeze(1)
            y = y.cuda()
            # calculate loss, add to test_loss, compute # correct predictions
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # normalizes test_loss by number of batches and calcules accuracy as percentage of correct predictions
    test_loss /= num_batches
    correct /= size
    wandb.log({"test_loss": test_loss, "test_accuracy": 100*correct, "epoch": epoch})
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml", entity="rczhang")
    best_loss = float('inf')
    patience = 10 # number of epochs to wait for improvement before stopping
    patience_counter = 0

    power, logpower, labels, freq = load_data('tessOBAstars.h5')
    learning_rate = 5e-4
    batch_size = 64
    epochs = 500
    num_channels = 32  # number of channels in first conv layer
    train_dataloader, test_dataloader, label_to_int, class_weights = preprocess_data(power, logpower, labels, freq, batch_size)
    input_size = len(power.iloc[0]) 
    model = CNN1D(num_channels, len(label_to_int)).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        current_loss = test_loop(test_dataloader, model, loss_fn, t)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    print("Done!")
    wandb.finish()