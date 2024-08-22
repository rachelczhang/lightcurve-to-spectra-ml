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
from sklearn.metrics import mean_squared_error, r2_score
import run_cnn
from data import read_hdf5_data

torch.manual_seed(42)
np.random.seed(42)

def read_data():
    df_iacob1 = pd.read_csv('/mnt/sdceph/users/rzhang/iacob1.csv')
    tic_id = df_iacob1['TIC_ID']
    teff = df_iacob1['Teff']
    logg = df_iacob1['logg']
    Msp = df_iacob1['Msp']
    return tic_id, teff, logg, Msp

def save_power_freq_info(h5_file_path):
    tic_id, teff, logg, Msp = read_data()
    db = read_tess_data.Database('/mnt/home/neisner/ceph/latte/output_LATTE/tess_database.db')
    with h5py.File(h5_file_path, 'w') as h5f:
        for tic_id, t, g, m in zip(tic_id, teff, logg, Msp):
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
                        h5f[dataset_name].attrs['logg'] = g
                        h5f[dataset_name].attrs['Msp'] = m
                        obs_id += 1

# def read_hdf5_data(hdf5_path):
#     power = []
#     Teff = []
#     logg = []
#     Msp = []
#     tic_id = []
#     with h5py.File(hdf5_path, 'r') as h5f:
#         if 'Frequency' in h5f:
#             frequencies = np.array(h5f['Frequency'])
#         for name in h5f:
#             if name != 'Frequency': 
#                 dataset = h5f[name]
#                 if not any('>' in str(dataset.attrs[attr]) or '<' in str(dataset.attrs[attr]) for attr in ['Teff', 'logg', 'Msp']):
#                     power.append(list(dataset))
#                     Teff.append(dataset.attrs['Teff'])
#                     logg.append(dataset.attrs['logg'])
#                     Msp.append(dataset.attrs['Msp'])
#                     tic_id.append(dataset.attrs['TIC_ID'])
#     power = pd.Series(power)
#     return power, Teff, logg, Msp, frequencies, tic_id

# def preprocess_data(power, Teff, logg, Msp, freq):
#     Teff = [float(t) for t in Teff]
#     logg = [float(l) for l in logg]
#     Msp = [float(m) for m in Msp]
#     logpower = power.apply(lambda x: np.log10(x).tolist())
#     scaled_power = apply_min_max_scaling(logpower)
#     # scaled_power = power
#     # convert from Series --> list of lists --> array --> tensor
#     power_tensor = torch.tensor(np.array(scaled_power.tolist(), dtype=np.float32))
#     # create labels tensor for Teff, logg    
#     labels_tensor = torch.tensor(list(zip(Teff, logg, Msp)), dtype=torch.float32)
#     return power_tensor, labels_tensor

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

def preprocess_data(power, Teff, logg, Msp, freq):
    Teff = [float(t) for t in Teff]
    logg = [float(l) for l in logg]
    Msp = [float(m) for m in Msp]
    
    # if tensor == False:
    Teff_norm, Teff_mean, Teff_std = normalize_data(np.array(Teff))
    logg_norm, logg_mean, logg_std = normalize_data(np.array(logg))
    Msp_norm, Msp_mean, Msp_std = normalize_data(np.array(Msp))
    # else:
    #     Teff_mean, Teff_std = torch.tensor(Teff_mean, dtype=torch.float64), torch.tensor(Teff_std, dtype=torch.float64)
    #     logg_mean, logg_std = torch.tensor(logg_mean, dtype=torch.float64), torch.tensor(logg_std, dtype=torch.float64)
    #     Msp_mean, Msp_std = torch.tensor(Msp_mean, dtype=torch.float64), torch.tensor(Msp_std, dtype=torch.float64)

    # log-transform and scale the power spectrum
    logpower = power.apply(lambda x: np.log10(x).tolist())
    scaled_power = apply_min_max_scaling(logpower)
    power_tensor = torch.tensor(np.array(scaled_power.tolist(), dtype=np.float32))
    
    # normalized labels tensor for Teff, logg, and Msp
    labels_tensor = torch.tensor(list(zip(Teff_norm, logg_norm, Msp_norm)), dtype=torch.float32)
    
    return power_tensor, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std)

def create_dataloaders(power_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

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

def calculate_lum_from_teff_logg(Teff, logg, Msp, tensor):
    if tensor == False:
        G = 6.67e-8
        sigma = 5.67e-5
        g_in_solar_mass = 1.989e33
        erg_in_solar_lum = 3.826e33
        g = 10 ** logg
        Msp_cgs = Msp * g_in_solar_mass
        Teff_K = Teff * 1000
        L = 4 * np.pi * G * Msp_cgs * sigma * Teff_K**4 / g
        L_solar = L / erg_in_solar_lum
        logL_solar = np.log10(L_solar)
    else:
        G = torch.tensor(6.67e-8, dtype=torch.float64)
        sigma = torch.tensor(5.67e-5, dtype=torch.float64)
        g_in_solar_mass = torch.tensor(1.989e33, dtype=torch.float64)
        erg_in_solar_lum = torch.tensor(3.826e33, dtype=torch.float64)
        g = 10 ** logg
        print('g', g)
        Msp_cgs = Msp * g_in_solar_mass
        print('Msp cgs', Msp_cgs)
        Teff_K = Teff * 1000
        print('Teff K', Teff_K)
        L = 4 * torch.pi * G * Msp_cgs * sigma * Teff_K**4 / g
        print('L', L)
        L_solar = L / erg_in_solar_lum
        print('L solar', L_solar)
        logL_solar = torch.log10(L_solar)
        print('logL solar', logL_solar)
    return logL_solar

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

def train_loop(dataloader, model, loss_fn, optimizer, epoch, norm_params):
    model.train()
    total_loss = 0
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std = norm_params
    for X, y in dataloader:
        X, y = X.cuda().unsqueeze(1), y.cuda() #UNSQUEEZE IF CNN
        print('epoch in train loop: ', epoch)
        pred = model(X)        
        # apply exponential transformation
        pred_exp = torch.exp(pred)

        Teff_pred, logg_pred, Msp_pred = pred_exp[:, 0].cpu() * Teff_std + Teff_mean, pred_exp[:, 1].cpu() * logg_std + logg_mean, pred_exp[:, 2].cpu() * Msp_std + Msp_mean
        Teff_actual, logg_actual, Msp_actual = y[:, 0].cpu() * Teff_std + Teff_mean, y[:, 1].cpu() * logg_std + logg_mean, y[:, 2].cpu() * Msp_std + Msp_mean
        
        Teff_pred = Teff_pred.to(dtype=torch.float64)
        logg_pred = logg_pred.to(dtype=torch.float64)
        Msp_pred = Msp_pred.to(dtype=torch.float64)
        Teff_actual = Teff_actual.to(dtype=torch.float64)
        logg_actual = logg_actual.to(dtype=torch.float64)
        Msp_actual = Msp_actual.to(dtype=torch.float64)

        pred_logL = calculate_lum_from_teff_logg(Teff_pred, logg_pred, Msp_pred, True).unsqueeze(-1).cuda()
        actual_logL = calculate_lum_from_teff_logg(Teff_actual, logg_actual, Msp_actual, True).unsqueeze(-1).cuda()

        loss = loss_fn(pred_logL, actual_logL)
        # loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f'Training loss: {avg_loss}')

def denormalize_data(norm_data, means, stds):
    denorm_data = []
    for data in norm_data:
        denorm_sublist = [(value * std + mean) for value, mean, std in zip(data, means, stds)]
        denorm_data.append(denorm_sublist)
    return np.array(denorm_data)

def test_loop(dataloader, model, loss_fn, epoch, norm_params):
    model.eval()
    total_loss = 0
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std = norm_params
    all_preds = []
    all_ys = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda().unsqueeze(1), y.cuda()
            pred = model(X)
            pred_exp = torch.exp(pred)
            Teff_pred, logg_pred, Msp_pred = pred_exp[:, 0].cpu() * Teff_std + Teff_mean, pred_exp[:, 1].cpu() * logg_std + logg_mean, pred_exp[:, 2].cpu() * Msp_std + Msp_mean
            Teff_actual, logg_actual, Msp_actual = y[:, 0].cpu() * Teff_std + Teff_mean, y[:, 1].cpu() * logg_std + logg_mean, y[:, 2].cpu() * Msp_std + Msp_mean
            Teff_pred = Teff_pred.to(dtype=torch.float64)
            logg_pred = logg_pred.to(dtype=torch.float64)
            Msp_pred = Msp_pred.to(dtype=torch.float64)
            Teff_actual = Teff_actual.to(dtype=torch.float64)
            logg_actual = logg_actual.to(dtype=torch.float64)
            Msp_actual = Msp_actual.to(dtype=torch.float64)
            pred_logL = calculate_lum_from_teff_logg(Teff_pred, logg_pred, Msp_pred, True).unsqueeze(-1).cuda()
            actual_logL = calculate_lum_from_teff_logg(Teff_actual, logg_actual, Msp_actual, True).unsqueeze(-1).cuda()
            total_loss += loss_fn(pred_logL, actual_logL)
            # total_loss += loss_fn(pred, y).item()
            all_preds.extend(pred_exp.cpu().numpy())
            all_ys.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    wandb.log({"test_loss": avg_loss, "epoch": epoch})
    print(f'Average loss: {avg_loss}')

    all_preds = denormalize_data(all_preds, [Teff_mean, logg_mean, Msp_mean], [Teff_std, logg_std, Msp_std])
    all_ys = denormalize_data(all_ys, [Teff_mean, logg_mean, Msp_mean], [Teff_std, logg_std, Msp_std])

    print('epoch in test loop: ', epoch)
    print('all ys', all_ys)
    print('all preds', all_preds)
    actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), np.array(all_ys[:, 2], dtype=np.float64), False)
    print('actual logL', actual_logL)
    pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), np.array(all_preds[:, 2], dtype=np.float64), False)
    print('pred logL', pred_logL)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 0], all_preds[:, 0], alpha=0.3)
    plt.xlabel('Actual Teff')
    plt.ylabel('Predicted Teff')
    plt.title('Predicted vs Actual Teff')
    plt.plot([all_ys[:, 0].min(), all_ys[:, 0].max()], [all_ys[:, 0].min(), all_ys[:, 0].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_Teff.png")
    wandb.log({"CNN Predicted vs Actual Teff": wandb.Image("pred_vs_act_Teff.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
    plt.xlabel('Actual logg')
    plt.ylabel('Predicted logg')
    plt.title('Predicted vs Actual logg')
    plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_logg.png")
    wandb.log({"CNN Predicted vs Actual logg": wandb.Image("pred_vs_act_logg.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_ys[:, 2], all_preds[:, 2], alpha=0.3)
    plt.xlabel('Actual Msp')
    plt.ylabel('Predicted Msp')
    plt.title('Predicted vs Actual Msp')
    plt.plot([all_ys[:, 2].min(), all_ys[:, 2].max()], [all_ys[:, 2].min(), all_ys[:, 2].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_Msp.png")
    wandb.log({"CNN Predicted vs Actual Msp": wandb.Image("pred_vs_act_Msp.png", caption="Predictions vs. Actual Msp at Epoch {}".format(epoch))})
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_logL, pred_logL, alpha=0.3)
    plt.xlabel('Actual logL')
    plt.ylabel('Predicted logL')
    plt.title('Predicted vs Actual Spectroscopic logL')
    plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_logL.png")
    wandb.log({"CNN Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
    plt.close()

    if ~np.isnan(pred_logL).any():
        mse = mean_squared_error(actual_logL, pred_logL)
        print('MSE', mse)
        
        r2 = r2_score(actual_logL, pred_logL)
        print('R2 score', r2)

    # residuals = all_preds[:, 0] - all_ys[:, 0]
    # plt.figure(figsize=(10, 6))
    # plt.scatter(all_ys[:, 0], residuals, alpha=0.3)
    # plt.xlabel('Actual Teff')
    # plt.ylabel('Residuals')
    # plt.title('Residuals Plot Teff')
    # plt.hlines(y=0, xmin=all_ys[:, 0].min(), xmax=all_ys[:, 0].max(), colors='r')
    # plt.grid(True)
    # plt.savefig("residuals_Teff.png")
    # wandb.log({"Residuals Teff": wandb.Image("residuals_Teff.png", caption="Residuals Teff at Epoch {}".format(epoch))})
    # plt.close()

    # residuals = all_preds[:, 1] - all_ys[:, 1]
    # plt.figure(figsize=(10, 6))
    # plt.scatter(all_ys[:, 1], residuals, alpha=0.3)
    # plt.xlabel('Actual logL')
    # plt.ylabel('Residuals')
    # plt.title('Residuals Plot logL')
    # plt.hlines(y=0, xmin=all_ys[:, 1].min(), xmax=all_ys[:, 1].max(), colors='r')
    # plt.grid(True)
    # plt.savefig("residuals_logL.png")
    # wandb.log({"Residuals logL": wandb.Image("residuals_logL.png", caption="Residuals logL at Epoch {}".format(epoch))})
    # plt.close()
    return avg_loss

# def get_spectroscopic_lum_info(all_ys, all_preds):
#     print('all ys', all_ys)
#     print('all preds', all_preds)
#     actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), np.array(all_ys[:, 2], dtype=np.float64))
#     print('actual logL', actual_logL)
#     pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), np.array(all_preds[:, 2], dtype=np.float64))
#     print('pred logL', pred_logL)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(actual_logL, pred_logL, alpha=0.3)
#     plt.xlabel('Actual logL')
#     plt.ylabel('Predicted logL')
#     plt.title('Predicted vs Actual Spectroscopic logL')
#     plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
#     plt.grid(True)
#     plt.savefig("pred_vs_act_logL.png")
#     plt.close()

#     mse = mean_squared_error(actual_logL, pred_logL)
#     print('MSE', mse)
    
#     r2 = r2_score(actual_logL, pred_logL)
#     print('R2 score', r2)

if __name__ == '__main__':
    wandb.init(project="lightcurve-to-spectra-ml-regression", entity="rczhang")
    h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression.h5'
    # save_power_freq_info(h5_file_path)
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')  
    print('len teff', len(Teff))
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, frequencies)
    print('normalization params', normalization_params)
    learning_rate = 1e-3
    batch_size = 32
    epochs = 10000
    train_loader, test_loader, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size)
    loss_fn = nn.MSELoss().cuda()
    # model = MLP(input_size=len(power.iloc[0]), output_size=3).cuda()
    num_channels = 32
    input_size = len(power.iloc[0])
    model = CNN1D(num_channels, 3, input_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
    best_loss = float('inf')
    patience = 200 
    patience_counter = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, t, normalization_params)
        current_loss = test_loop(test_loader, model, loss_fn, t, normalization_params)
        scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_regression_cnn.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break 
    # get_spectroscopic_lum_info(all_ys, all_preds)
    print("Done!")
    wandb.finish()