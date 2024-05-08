import torch
import torch.nn as nn
from run_mlp import load_data, preprocess_data
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np 

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
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f"Train loss: {avg_loss:>7f}")

def compute_confusion_matrix(true, pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        conf_matrix[t, p] += 1
    return conf_matrix

def test_loop(dataloader, model, loss_fn, epoch, num_classes=3):
    """
    Evaluate the model's performance on the test dataset
    """
    # put model in evaluation mode
    model.eval()
    # iniitalizes variables to accumulate total loss and number of correctly predicted samples
    test_loss, correct = 0, 0
    all_labels = []
    all_predictions = []
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
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(pred.argmax(1).cpu().numpy())
    # calculate confusion matrix
    conf_matrix = compute_confusion_matrix(all_labels, all_predictions, num_classes) 
    # calculates average test loss per batch and calcules accuracy as percentage of correct predictions
    avg_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset) # len(dataloader.dataset) = number of samples in dataset
    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy, "epoch": epoch, "predictions_histogram": wandb.Histogram(all_predictions)})
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    counts, _ = np.histogram(all_predictions, bins=np.arange(-0.5, len(label_to_int) + 0.5))
    print(f"Counts per class: {counts}")
    print(f"Confusion matrix: \n{conf_matrix}")
    return avg_loss

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml", entity="rczhang")
    best_loss = float('inf')
    patience = 500 # number of epochs to wait for improvement before stopping
    patience_counter = 0

    power, logpower, labels, freq = load_data('tessOBAstars.h5')
    learning_rate = 1e-3
    batch_size = 64
    epochs = 500
    num_channels = 32  # number of channels in first conv layer
    train_dataloader, test_dataloader, label_to_int, class_weights = preprocess_data(power, logpower, labels, freq, batch_size)
    input_size = len(power.iloc[0]) 
    model = CNN1D(num_channels, len(label_to_int)).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        current_loss = test_loop(test_dataloader, model, loss_fn, t)
        scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_cnn.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    print("Done!")
    wandb.finish()