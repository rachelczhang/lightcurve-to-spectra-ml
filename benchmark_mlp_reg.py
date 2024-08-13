import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import mean_squared_error, r2_score

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
    Teff = data['Teff']
    logg = data['logg']
    Msp = data['Msp']
    return alpha0, nu_char, gamma, Cw, Teff, logg, Msp

def preprocess_data(alpha0, nu_char, gamma, Cw, Teff, logg, Msp):
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

    Teff = [float(t) for t in Teff]
    logg = [float(l) for l in logg]
    Msp = [float(m) for m in Msp]

    labels_tensor = torch.tensor(list(zip(Teff, logg, Msp)), dtype=torch.float32)

    return data_normalized, labels_tensor

def create_dataloaders(power_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

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

def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_ys = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            all_preds.extend(pred.cpu().numpy())
            all_ys.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    wandb.log({"test_loss": avg_loss, "epoch": epoch})
    print(f'Average loss: {avg_loss}')

    all_preds = np.array(all_preds)
    all_ys = np.array(all_ys)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 0], all_preds[:, 0], alpha=0.3)
    plt.xlabel('Actual Teff')
    plt.ylabel('Predicted Teff')
    plt.title('Predicted vs Actual Teff')
    plt.plot([all_ys[:, 0].min(), all_ys[:, 0].max()], [all_ys[:, 0].min(), all_ys[:, 0].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_Teff.png")
    wandb.log({"MLP Predicted vs Actual Teff": wandb.Image("pred_vs_act_Teff.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
    plt.xlabel('Actual logg')
    plt.ylabel('Predicted logg')
    plt.title('Predicted vs Actual logg')
    plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_logg.png")
    wandb.log({"MLP Predicted vs Actual logg": wandb.Image("pred_vs_act_logg.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 2], all_preds[:, 2], alpha=0.3)
    plt.xlabel('Actual Msp')
    plt.ylabel('Predicted Msp')
    plt.title('Predicted vs Actual Msp')
    plt.plot([all_ys[:, 2].min(), all_ys[:, 2].max()], [all_ys[:, 2].min(), all_ys[:, 2].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_Msp.png")
    wandb.log({"MLP Predicted vs Actual Msp": wandb.Image("pred_vs_act_Msp.png", caption="Predictions vs. Actual Msp at Epoch {}".format(epoch))})
    plt.close()

    print('all ys', all_ys)
    print('all preds', all_preds)
    actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), np.array(all_ys[:, 2], dtype=np.float64))
    print('actual logL', actual_logL)
    pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), np.array(all_preds[:, 2], dtype=np.float64))
    print('pred logL', pred_logL)
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_logL, pred_logL, alpha=0.3)
    plt.xlabel('Actual logL')
    plt.ylabel('Predicted logL')
    plt.title('Predicted vs Actual Spectroscopic logL')
    plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_logL.png")
    wandb.log({"MLP Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
    plt.close()

    mse = mean_squared_error(actual_logL, pred_logL)
    print('MSE at Epoch ', epoch, ': ', mse)
    r2 = r2_score(actual_logL, pred_logL)
    print('R2 score at Epoch ', epoch, ': ', r2)

    return avg_loss, all_ys, all_preds

def calculate_lum_from_teff_logg(Teff, logg, Msp):
    # cgs units
    G = np.float64(6.67e-8)
    sigma = np.float64(5.67e-5)
    g = 10**logg
    Msp = Msp * np.float64(1.989e33)
    Teff_K = Teff*10**3
    L = 4 * np.pi * G * Msp * sigma * Teff_K**4 / g
    L_solar = L/np.float64(3.826e33)
    logL_solar = np.log10(L_solar)
    return logL_solar

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-benchmark-mlp-reg", entity="rczhang")
    best_loss = float('inf')
    patience = 200 
    patience_counter = 0	
    alpha0, nu_char, gamma, Cw, Teff, logg, Msp = load_data('curvefitparams_reg.h5')
    epochs = 10000
    batch_size = 32
    data_normalized, labels_tensor = preprocess_data(alpha0, nu_char, gamma, Cw, Teff, logg, Msp)
    print('data normalized', data_normalized)
    train_loader, test_loader, test_dataset = create_dataloaders(data_normalized, labels_tensor, batch_size)
    loss_fn = nn.MSELoss().cuda()
    model = MLP(input_size=4, output_size=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, t)
        current_loss, all_ys, all_preds = test_loop(test_loader, model, loss_fn, t)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_benchmark_mlp_reg.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    print("Done!")
    wandb.finish()