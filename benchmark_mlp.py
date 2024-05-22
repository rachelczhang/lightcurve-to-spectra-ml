import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from run_mlp import encode_labels, calculate_class_weights
from run_cnn import compute_confusion_matrix
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay

torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            # nn.Softmax()
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def load_data(filename):
	""" directly read the data from curvefitparams.h5 """
	data = pd.read_hdf(filename)
	alpha0 = data['alpha0']
	nu_char = data['nu_char']
	gamma = data['gamma']
	Cw = data['Cw']
	labels = data['labels']
	return alpha0, nu_char, gamma, Cw, labels

def preprocess_data(alpha0, nu_char, gamma, Cw, labels):
	""" convert data to tensors and apply normalization."""
	alpha0 = torch.tensor(alpha0.values, dtype=torch.float32)
	nu_char = torch.tensor(nu_char.values, dtype=torch.float32)
	gamma = torch.tensor(gamma.values, dtype=torch.float32)
	Cw = torch.tensor(Cw.values, dtype=torch.float32)

	data = torch.stack([alpha0, nu_char, gamma, Cw], dim=1)

	min_vals = torch.min(data, dim=0).values
	max_vals = torch.max(data, dim=0).values

	range_vals = max_vals - min_vals
	data_normalized = (data - min_vals) / range_vals

	labels_tensor, label_to_int = encode_labels(labels)

	return data_normalized, labels_tensor, label_to_int

def create_dataloaders(data, labels, batch_size=32, test_size=0.2, random_state=42):
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)

	train_dataset = TensorDataset(X_train, y_train)
	test_dataset = TensorDataset(X_test, y_test)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	class_weights = calculate_class_weights(labels)
	return train_loader, test_loader, class_weights

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda()
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

def test_loop(dataloader, model, loss_fn, epoch, num_classes=3):
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
            X = X.cuda()
            y = y.cuda()
            # calculate loss, add to test_loss, compute # correct predictions
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_labels.extend(y.cpu().numpy())
            all_probabilities.append(nn.functional.softmax(pred, dim=1).cpu())
            all_predictions.extend(pred.argmax(1).cpu().numpy())
    all_probabilities = np.vstack(all_probabilities)
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
	wandb.init(project="lightcurve-to-spectra-ml-benchmark-mlp", entity="rczhang")
	alpha0, nu_char, gamma, Cw, labels = load_data('curvefitparams.h5')
	learning_rate = 1e-4
	epochs = 100
	data_normalized, labels, label_to_int = preprocess_data(alpha0, nu_char, gamma, Cw, labels)
	print('data normalized', data_normalized)
	train_loader, test_loader, class_weights = create_dataloaders(data_normalized, labels)
	loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()#, reduction='sum')
	model = MLP(input_size=4, output_size=len(label_to_int)).cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(train_loader, model, loss_fn, optimizer, t)
		test_loop(test_loader, model, loss_fn, t)
	print("Done!")
	wandb.finish()