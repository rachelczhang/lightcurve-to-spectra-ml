import torch
import torch.nn as nn
from run_mlp import load_data, preprocess_data, createdataloaders
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np 
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
# def undersample_data(X, y):
#     # convert labels tensor to numpy array to utilize numpy operations
#     y_np = y.numpy()
#     # gets the unique values in the labels array and the count of each unique value
#     unique, counts = np.unique(y_np, return_counts=True)
#     # gets the minimum count of any class
#     min_count = np.min(counts)
#     # list to store indices to keep
#     indices_to_keep = []
#     # in this case, class_value is 0, 1, then 2
#     for class_value in unique:
#         # get all the indices of the y_np array where the value is the class_value
#         class_indices = np.where(y_np == class_value)[0]
#         # randomly sample min_count number of indices from the class_indices array
#         undersampled_indices = np.random.choice(class_indices, min_count, replace=False)
#         indices_to_keep.append(undersampled_indices)
#     # combine the three arrays for class_value 0, 1, and 2 into one array
#     indices_to_keep = np.concatenate(indices_to_keep)
#     # shuffle the indices so the model gets a generalizable batch for training
#     np.random.shuffle(indices_to_keep)
#     # extract the corresponding X and y values for the indices to keep
#     X_undersampled = X[indices_to_keep]
#     y_undersampled = y[indices_to_keep]
#     return X_undersampled, y_undersampled


############## THIS IS THE BEST WORKING CNN MODEL ################
# class CNN1D(nn.Module):
#     def __init__(self, num_channels, output_size, input_size):
#         """
#         num_channels: # of output channels for first convolutional layer
#         output_size: # of classes for the final output layer
#         """
#         super().__init__()
#         self.conv_layers = nn.Sequential( # container of layers that processes convolutional part of network
#             nn.Conv1d(1, num_channels, kernel_size=5, padding=2),  # Convolutional layer with 1 input channel, num_channels output channels, 5 length kernel, and 2 elements of padding on either side
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2), # max pooling layer that reduces dimensionality by taking max value of each 2-element window
#             nn.Conv1d(num_channels, num_channels * 2, kernel_size=5, padding=2),
#             # nn.BatchNorm1d(num_channels * 2), 
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             # nn.MaxPool1d(kernel_size=2), # 06/07: TESTING ONE MORE POOLING LAYER
#         )
#         # use a dummy input to dynamically determine the output dimension
#         dummy_input = torch.randn(1, 1, input_size)  # batch size of 1, 1 channel, and initial input size
#         dummy_output = self.conv_layers(dummy_input)
#         self.output_dim = dummy_output.numel() // dummy_output.shape[0]  # total number of features divided by batch size
#         print('output dim', self.output_dim)
#         print('hard coded dim', num_channels * 2 * (input_size // 4))
#         self.flatten = nn.Flatten() # flattens multiple-dimensional tensor convolutional layers output --> 1D tensor
#         self.fc_layers = nn.Sequential( # container of layers that processes fully connected part of network
#             nn.Linear(self.output_dim, 128),  # nn.Linear(num_channels * 2 * (input_size // 4), 128), 
#             nn.ReLU(),
#             nn.Linear(128, output_size),
#         )
    
#     def forward(self, x):
#         x = self.conv_layers(x) # process inputs through convolutional layers
#         x = self.flatten(x) # flattens output of convolutional layers
#         logits = self.fc_layers(x) # passes flattened output through fully connected layers to produce logits for each class
#         return logits # raw, unnormalized scores for each class

############### TESTING OTHER CNN MODELS TO COMPARE WITH PRE-TRAINING ################
class CNN1D(nn.Module):
    def __init__(self, num_channels, output_size, input_size):
        """
        num_channels: # of output channels for first convolutional layer
        output_size: # of classes for the final output layer
        """
        super().__init__()
        self.conv_layers = nn.Sequential( # container of layers that processes convolutional part of network
            nn.Conv1d(1, num_channels, kernel_size=5, padding=2),  # Convolutional layer with 1 input channel, num_channels output channels, 5 length kernel, and 2 elements of padding on either side
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # max pooling layer that reduces dimensionality by taking max value of each 2-element window
            nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
            # nn.BatchNorm1d(num_channels * 2), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2)
            # nn.MaxPool1d(kernel_size=2), # 06/07: TESTING ONE MORE POOLING LAYER
        )
        # use a dummy input to dynamically determine the output dimension
        dummy_input = torch.randn(1, 1, input_size)  # batch size of 1, 1 channel, and initial input size
        dummy_output = self.conv_layers(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]  # total number of features divided by batch size
        print('output dim', self.output_dim)
        print('hard coded dim', num_channels * 2 * (input_size // 4))
        self.flatten = nn.Flatten() # flattens multiple-dimensional tensor convolutional layers output --> 1D tensor
        self.fc_layers = nn.Sequential( # container of layers that processes fully connected part of network
            nn.Linear(self.output_dim, 128),  # nn.Linear(num_channels * 2 * (input_size // 4), 128), 
            nn.ReLU(),
            # nn.Linear(128, 64),  
            # nn.ReLU(),
            nn.Linear(128, output_size),
        )
    
    def forward(self, x):
        x = self.conv_layers(x) # process inputs through convolutional layers
        x = self.flatten(x) # flattens output of convolutional layers
        logits = self.fc_layers(x) # passes flattened output through fully connected layers to produce logits for each class
        return logits # raw, unnormalized scores for each class

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
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

def test_loop(dataloader, model, loss_fn, epoch, label_to_int, num_classes=3):
    """
    Evaluate the model's performance on the test dataset
    """
    # put model in evaluation mode
    model.eval()
    # iniitalizes variables to accumulate total loss and number of correctly predicted samples
    test_loss, correct = 0, 0
    all_labels = []
    all_probabilities = []
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
            all_probabilities.append(nn.functional.softmax(pred, dim=1).cpu())
            all_predictions.extend(pred.argmax(1).cpu().numpy())
    all_probabilities = np.vstack(all_probabilities)
    
    ######### Evaluation Metrics ################

    # calculate confusion matrix
    conf_matrix = compute_confusion_matrix(all_labels, all_predictions, num_classes) 
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=list(label_to_int.keys()))
    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(values_format='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix at Epoch {}".format(epoch))
    plt.tight_layout()
    plt.savefig("conf_matrix.png")
    plt.close(fig)
    wandb.log({"conf_matrix": wandb.Image("conf_matrix.png", caption="Confusion Matrix at Epoch {}".format(epoch))})

    # calculates average test loss per batch and calcules accuracy as percentage of correct predictions
    avg_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset) # len(dataloader.dataset) = number of samples in dataset
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    wandb.log({
        "test_loss": avg_loss, "test_accuracy": accuracy, "epoch": epoch, "predictions_histogram": wandb.Histogram(all_predictions),
        "precision": precision, "recall": recall, "auc": auc, "balanced_accuracy": balanced_acc, "mcc": mcc})
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    counts, _ = np.histogram(all_predictions, bins=np.arange(-0.5, len(label_to_int) + 0.5))
    print(f"Counts per class: {counts}")
    print(f"Confusion matrix: \n{conf_matrix}")
    return avg_loss

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-cnn", entity="rczhang")
    best_loss = float('inf')
    patience = 200 # number of epochs to wait for improvement before stopping
    patience_counter = 0

    power, logpower, labels, freq = load_data('tessOBAstars.h5')
    learning_rate = 1e-3
    batch_size = 64
    epochs = 500
    num_channels = 32  # number of channels in first conv layer
    power_tensor, labels_tensor, label_to_int = preprocess_data(power, logpower, labels, freq)
    # power_tensor_undersampled, labels_tensor_undersampled = undersample_data(power_tensor, labels_tensor)
    train_dataloader, test_dataloader, class_weights = createdataloaders(power_tensor, labels_tensor, batch_size, augment_data=True, additive=False)
    input_size = len(power.iloc[0]) 
    model = CNN1D(num_channels, len(label_to_int), input_size).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()
    t = -1
    current_loss = test_loop(test_dataloader, model, loss_fn, t, label_to_int)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        current_loss = test_loop(test_dataloader, model, loss_fn, t, label_to_int)
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