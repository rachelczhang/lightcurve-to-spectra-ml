import pandas as pd 
import read_tess_data
from read_tess_data import read_light_curve, light_curve_to_power_spectrum
import h5py
import numpy as np
from run_mlp import apply_min_max_scaling
import torch 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def read_data():
    df_iacob1 = pd.read_csv('/mnt/sdceph/users/rzhang/iacob1.csv')
    tic_id = df_iacob1['TIC_ID']
    teff = df_iacob1['Teff']
    logL = df_iacob1['logL']
    return tic_id, teff, logL

def save_power_freq_info():
    tic_id, teff, logL = read_data()
    db = read_tess_data.Database('/mnt/home/neisner/ceph/latte/output_LATTE/tess_database.db')
    h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression.h5'
    with h5py.File(h5_file_path, 'w') as h5f:
        for tic_id, t, l in zip(tic_id, teff, logL):
            sectorids, lcpaths, tppaths = db.search(tic_id)
            if lcpaths != 0:
                obs_id = 0
                for filepath in lcpaths:
                    print('looking for filepath: ', filepath)
                    if read_light_curve(filepath) is not None:
                        time, flux = read_light_curve(filepath)
                        freq, power = light_curve_to_power_spectrum(time, flux)
                        dataset_name = f'TIC_{tic_id}_{obs_id}_Power'
                        h5f.create_dataset(dataset_name, data=power)
                        if 'Frequency' not in h5f:
                            h5f.create_dataset('Frequency', data=freq)
                        h5f[dataset_name].attrs['TIC_ID'] = tic_id
                        h5f[dataset_name].attrs['Teff'] = t
                        h5f[dataset_name].attrs['logL'] = l
                        obs_id += 1

def read_hdf5_data(hdf5_path):
    power = []
    Teff = []
    logL = []
    tic_id = []
    with h5py.File(hdf5_path, 'r') as h5f:
        if 'Frequency' in h5f:
            frequencies = np.array(h5f['Frequency'])
        for name in h5f:
            if name != 'Frequency': 
                dataset = h5f[name]
                power.append(list(dataset))
                Teff.append(dataset.attrs['Teff'])
                logL.append(dataset.attrs['logL'])
                tic_id.append(dataset.attrs['TIC_ID'])
    power = pd.Series(power)
    return power, Teff, logL, frequencies, tic_id

def preprocess_data(power, Teff, logL, freq):
    Teff = [float(t) for t in Teff]
    logL = [float(l) for l in logL]
    logpower = power.apply(lambda x: np.log10(x).tolist())
    scaled_power = apply_min_max_scaling(logpower)
    # convert from Series --> list of lists --> array --> tensor
    power_tensor = torch.tensor(np.array(scaled_power.tolist(), dtype=np.float32))
    # create labels tensor for Teff, logL    
    labels_tensor = torch.tensor(list(zip(Teff, logL)), dtype=torch.float32)
    return power_tensor, labels_tensor

def create_dataloaders(power_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

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
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.cuda(), y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f'Training loss: {avg_loss}')

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
    wandb.log({"Predicted vs Actual Teff": wandb.Image("pred_vs_act_Teff.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
    plt.xlabel('Actual logL')
    plt.ylabel('Predicted logL')
    plt.title('Predicted vs Actual logL')
    plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_logL.png")
    wandb.log({"Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
    plt.close()

    residuals = all_preds[:, 0] - all_ys[:, 0]
    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 0], residuals, alpha=0.3)
    plt.xlabel('Actual Teff')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot Teff')
    plt.hlines(y=0, xmin=all_ys[:, 0].min(), xmax=all_ys[:, 0].max(), colors='r')
    plt.grid(True)
    plt.savefig("residuals_Teff.png")
    wandb.log({"Residuals Teff": wandb.Image("residuals_Teff.png", caption="Residuals Teff at Epoch {}".format(epoch))})
    plt.close()

    residuals = all_preds[:, 1] - all_ys[:, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 1], residuals, alpha=0.3)
    plt.xlabel('Actual logL')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot logL')
    plt.hlines(y=0, xmin=all_ys[:, 1].min(), xmax=all_ys[:, 1].max(), colors='r')
    plt.grid(True)
    plt.savefig("residuals_logL.png")
    wandb.log({"Residuals logL": wandb.Image("residuals_logL.png", caption="Residuals logL at Epoch {}".format(epoch))})
    plt.close()

    return avg_loss

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-regression", entity="rczhang")
    # save_power_freq_info()
    power, Teff, logL, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')  
    power_tensor, labels_tensor = preprocess_data(power, Teff, logL, frequencies)
    learning_rate = 1e-3
    batch_size = 32
    epochs = 10000
    train_loader, test_loader = create_dataloaders(power_tensor, labels_tensor, batch_size)
    loss_fn = nn.MSELoss().cuda()
    model = MLP(input_size=len(power.iloc[0]), output_size=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
    best_loss = float('inf')
    patience = 200 
    patience_counter = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, t)
        current_loss = test_loop(test_loader, model, loss_fn, t)
        scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_regression_mlp.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    print("Done!")
    wandb.finish()