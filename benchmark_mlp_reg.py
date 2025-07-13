import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# Set global seeds for reproducibility
def set_all_seeds(seed):
    """Set all random seeds for reproducible results"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For complete reproducibility (at cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

# def calculate_lum_from_teff_logg(Teff, logg, Msp):
#     # cgs units
#     G = np.float64(6.67e-8)
#     sigma = np.float64(5.67e-5)
#     g = 10**logg
#     Msp = Msp * np.float64(1.989e33)
#     Teff_K = Teff*10**3
#     L = 4 * np.pi * G * Msp * sigma * Teff_K**4 / g
#     L_solar = L/np.float64(3.826e33)
#     logL_solar = np.log10(L_solar)
#     return logL_solar

def calculate_lum_from_teff_logg(Teff, logg, tensor):
    if tensor == False:
        log_L_sun=np.log10((5777)**4.0/(274*100))  # Used for Spectroscopic HRD
        Teff_K = Teff * 1000
        logL_solar = 4*np.log10(Teff_K)-logg-log_L_sun
        # G = 6.67e-8
        # sigma = 5.67e-5
        # g_in_solar_mass = 1.989e33
        # erg_in_solar_lum = 3.826e33
        # g = 10 ** logg
        # # if np.isinf(g).any():
        # #     g[np.isinf(g)] = 1e6
        # Msp_cgs = Msp * g_in_solar_mass
        # Teff_K = Teff * 1000
        # L = 4 * np.pi * G * Msp_cgs * sigma * Teff_K**4 / g
        # L_solar = L / erg_in_solar_lum
        # logL_solar = np.log10(L_solar)
    else:
        log_L_sun=torch.tensor((np.log10((5777)**4.0/(274*100))), dtype=torch.float64)
        Teff_K = Teff * 1000
        logL_solar = 4*torch.log10(Teff_K)-logg-log_L_sun
        # G = torch.tensor(6.67e-8, dtype=torch.float64)
        # sigma = torch.tensor(5.67e-5, dtype=torch.float64)
        # g_in_solar_mass = torch.tensor(1.989e33, dtype=torch.float64)
        # erg_in_solar_lum = torch.tensor(3.826e33, dtype=torch.float64)
        # g = 10 ** logg
        # # print('g', g)
        # # if torch.isinf(g).any():
        # #     g[torch.isinf(g)] = torch.tensor(1e6, dtype=torch.float64)
        # Msp_cgs = Msp * g_in_solar_mass
        # Teff_K = Teff * 1000
        # L = 4 * torch.pi * G * Msp_cgs * sigma * Teff_K**4 / g
        # L_solar = L / erg_in_solar_lum
        # logL_solar = torch.log10(L_solar)
    return logL_solar

def preprocess_data(alpha0, nu_char, gamma, Cw, Teff, logg, Msp, predict_target='teff_logg'):
    """ convert data to tensors and apply normalization.
    predict_target: 'teff_logg' or 'teff_logl'
    """
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
    Teff_norm, Teff_mean, Teff_std = normalize_data(np.array(Teff))
    logg_norm, logg_mean, logg_std = normalize_data(np.array(logg))
    Msp_norm, Msp_mean, Msp_std = normalize_data(np.array(Msp))
    logL = calculate_lum_from_teff_logg(np.array(Teff), np.array(logg), False)
    logL_norm, logL_mean, logL_std = normalize_data(logL)

    # Create labels based on prediction target
    if predict_target == 'teff_logg':
        # # TEST ONLY TEFF
        # labels_tensor = torch.tensor(list(zip(Teff_norm)), dtype=torch.float32)
        # change between MLP1 and MLP2
        labels_tensor = torch.tensor(list(zip(Teff_norm, logg_norm)), dtype=torch.float32)
        # labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)
    elif predict_target == 'teff_logl':
        labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)
    else:
        raise ValueError("predict_target must be 'teff_logg' or 'teff_logl'")

    return data_normalized, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std)

def create_dataloaders(power_tensor, labels_tensor, batch_size, seed=42):
    """Create train/test dataloaders with deterministic splitting"""
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Use a generator with fixed seed for deterministic splitting
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    
    # Create generators for DataLoaders as well
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
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

def denormalize_data(norm_data, means, stds):
    denorm_data = []
    for data in norm_data:
        denorm_sublist = [(value * std + mean) for value, mean, std in zip(data, means, stds)]
        denorm_data.append(denorm_sublist)
    return np.array(denorm_data)

def test_loop(dataloader, model, loss_fn, epoch, norm_params, predict_target='teff_logg', output_details=False, print_loss=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_ys = []
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            all_preds.extend(pred.cpu().numpy())
            all_ys.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    wandb.log({"test_loss": avg_loss, "epoch": epoch})
    if print_loss:
        print(f'Average loss: {avg_loss}')

    all_preds = np.array(all_preds)
    all_ys = np.array(all_ys)

    # Denormalize based on prediction target
    if predict_target == 'teff_logg':
        # # TEST JUST TEFF
        # all_preds = denormalize_data(all_preds, [Teff_mean], [Teff_std])
        # all_ys = denormalize_data(all_ys, [Teff_mean], [Teff_std])

        all_preds = denormalize_data(all_preds, [Teff_mean, logg_mean], [Teff_std, logg_std])
        all_ys = denormalize_data(all_ys, [Teff_mean, logg_mean], [Teff_std, logg_std])
    elif predict_target == 'teff_logl':
        all_preds = denormalize_data(all_preds, [Teff_mean, logL_mean], [Teff_std, logL_std])
        all_ys = denormalize_data(all_ys, [Teff_mean, logL_mean], [Teff_std, logL_std])
    
    # Always create Teff plot for wandb (but conditionally save detailed output)
    if output_details:
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

    if predict_target == 'teff_logg':
        # Calculate derived logL
        actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), False)
        pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), False)
        
        # Create plots only if detailed output requested
        if output_details:
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
            plt.scatter(actual_logL, pred_logL, alpha=0.3)
            plt.xlabel('Actual logL')
            plt.ylabel('Predicted logL')
            plt.title('Predicted vs Actual Spectroscopic logL')
            plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logL.png")
            wandb.log({"MLP Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
            plt.close()

        # Calculate and print metrics
        r2_teff = r2_score(all_ys[:, 0], all_preds[:, 0])
        r2_logg = r2_score(all_ys[:, 1], all_preds[:, 1])
        r2_logl = r2_score(actual_logL, pred_logL)
        mse_teff = mean_squared_error(all_ys[:, 0], all_preds[:, 0])
        mse_logg = mean_squared_error(all_ys[:, 1], all_preds[:, 1])
        mse_logl = mean_squared_error(actual_logL, pred_logL)
        
        print(f'R2 score of Teff at Epoch {epoch}: {r2_teff:.6f}')
        print(f'R2 score of logg at Epoch {epoch}: {r2_logg:.6f}')
        print(f'R2 score of logL at Epoch {epoch}: {r2_logl:.6f}')
        print(f'MSE of Teff at Epoch {epoch}: {mse_teff:.6f}')
        print(f'MSE of logg at Epoch {epoch}: {mse_logg:.6f}')
        print(f'MSE of logL at Epoch {epoch}: {mse_logl:.6f}')
        
        # Output detailed arrays only if requested
        if output_details:
            print(f'\n{"="*60}')
            print(f'DETAILED ARRAYS FOR PLOTTING (Epoch {epoch})')
            print(f'{"="*60}')
            print('actual_Teff =', [f'{x:.6f}' for x in all_ys[:, 0]])
            print('pred_Teff =', [f'{x:.6f}' for x in all_preds[:, 0]])
            print('actual_logg =', [f'{x:.6f}' for x in all_ys[:, 1]])
            print('pred_logg =', [f'{x:.6f}' for x in all_preds[:, 1]])
            print('actual_logL =', [f'{x:.6f}' for x in actual_logL])
            print('pred_logL =', [f'{x:.6f}' for x in pred_logL])
            print(f'{"="*60}\n')
    
    elif predict_target == 'teff_logl':
        # For MLP2 - directly predicting logL
        if output_details:
            plt.figure(figsize=(10, 6))
            plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
            plt.xlabel('Actual logL')
            plt.ylabel('Predicted logL')
            plt.title('Predicted vs Actual logL (Direct)')
            plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logL_direct.png")
            wandb.log({"MLP Predicted vs Actual logL (Direct)": wandb.Image("pred_vs_act_logL_direct.png", caption="Predictions vs. Actual logL (Direct) at Epoch {}".format(epoch))})
            plt.close()

        # Calculate and print metrics
        r2_teff = r2_score(all_ys[:, 0], all_preds[:, 0])
        r2_logl = r2_score(all_ys[:, 1], all_preds[:, 1])
        mse_teff = mean_squared_error(all_ys[:, 0], all_preds[:, 0])
        mse_logl = mean_squared_error(all_ys[:, 1], all_preds[:, 1])
        
        print(f'R2 score of Teff at Epoch {epoch}: {r2_teff:.6f}')
        print(f'R2 score of logL (direct) at Epoch {epoch}: {r2_logl:.6f}')
        print(f'MSE of Teff at Epoch {epoch}: {mse_teff:.6f}')
        print(f'MSE of logL (direct) at Epoch {epoch}: {mse_logl:.6f}')
        
        # Output detailed arrays only if requested
        if output_details:
            print(f'\n{"="*60}')
            print(f'DETAILED ARRAYS FOR PLOTTING (Epoch {epoch})')
            print(f'{"="*60}')
            print('actual_Teff =', [f'{x:.6f}' for x in all_ys[:, 0]])
            print('pred_Teff =', [f'{x:.6f}' for x in all_preds[:, 0]])
            print('actual_logL =', [f'{x:.6f}' for x in all_ys[:, 1]])
            print('pred_logL =', [f'{x:.6f}' for x in all_preds[:, 1]])
            print(f'{"="*60}\n')

    return avg_loss, all_ys, all_preds

def final_evaluation_with_stored_results(all_actual_values, all_predictions, norm_params, predict_target, epoch_name):
    """
    This function performs the final evaluation using the stored best_predictions and best_actual_values.
    It also handles the detailed plotting and logging for the final evaluation.
    Note: all_actual_values and all_predictions are already denormalized from test_loop.
    """
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
   
    # Always create Teff plot for wandb (but conditionally save detailed output)
    if True: # Always plot for final evaluation
        plt.figure(figsize=(10, 6))
        plt.scatter(all_actual_values[:, 0], all_predictions[:, 0], alpha=0.3)
        plt.xlabel('Actual Teff')
        plt.ylabel('Predicted Teff')
        plt.title('Predicted vs Actual Teff')
        plt.plot([all_actual_values[:, 0].min(), all_actual_values[:, 0].max()], [all_actual_values[:, 0].min(), all_actual_values[:, 0].max()], 'r')
        plt.grid(True)
        plt.savefig("pred_vs_act_Teff_final.png")
        wandb.log({"MLP Predicted vs Actual Teff Final": wandb.Image("pred_vs_act_Teff_final.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch_name))})
        plt.close()

    if predict_target == 'teff_logg':
        # Calculate derived logL
        actual_logL = calculate_lum_from_teff_logg(np.array(all_actual_values[:, 0], dtype=np.float64), np.array(all_actual_values[:, 1], dtype=np.float64), False)
        pred_logL = calculate_lum_from_teff_logg(np.array(all_predictions[:, 0], dtype=np.float64), np.array(all_predictions[:, 1], dtype=np.float64), False)
        
        # Create plots only if detailed output requested
        if True: # Always plot for final evaluation
            plt.figure(figsize=(10, 6))
            plt.scatter(all_actual_values[:, 1], all_predictions[:, 1], alpha=0.3)
            plt.xlabel('Actual logg')
            plt.ylabel('Predicted logg')
            plt.title('Predicted vs Actual logg')
            plt.plot([all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], [all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logg_final.png")
            wandb.log({"MLP Predicted vs Actual logg Final": wandb.Image("pred_vs_act_logg_final.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch_name))})
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.scatter(actual_logL, pred_logL, alpha=0.3)
            plt.xlabel('Actual logL')
            plt.ylabel('Predicted logL')
            plt.title('Predicted vs Actual Spectroscopic logL')
            plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logL_final.png")
            wandb.log({"MLP Predicted vs Actual logL Final": wandb.Image("pred_vs_act_logL_final.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch_name))})
            plt.close()

        # Calculate and print metrics
        r2_teff = r2_score(all_actual_values[:, 0], all_predictions[:, 0])
        r2_logg = r2_score(all_actual_values[:, 1], all_predictions[:, 1])
        r2_logl = r2_score(actual_logL, pred_logL)
        mse_teff = mean_squared_error(all_actual_values[:, 0], all_predictions[:, 0])
        mse_logg = mean_squared_error(all_actual_values[:, 1], all_predictions[:, 1])
        mse_logl = mean_squared_error(actual_logL, pred_logL)
        
        print(f'R2 score of Teff at Epoch {epoch_name}: {r2_teff:.6f}')
        print(f'R2 score of logg at Epoch {epoch_name}: {r2_logg:.6f}')
        print(f'R2 score of logL at Epoch {epoch_name}: {r2_logl:.6f}')
        print(f'MSE of Teff at Epoch {epoch_name}: {mse_teff:.6f}')
        print(f'MSE of logg at Epoch {epoch_name}: {mse_logg:.6f}')
        print(f'MSE of logL at Epoch {epoch_name}: {mse_logl:.6f}')
        
        # Output detailed arrays only if requested
        if True: # Always print for final evaluation
            print(f'\n{"="*60}')
            print(f'DETAILED ARRAYS FOR PLOTTING (Epoch {epoch_name})')
            print(f'{"="*60}')
            print('actual_Teff =', [f'{x:.6f}' for x in all_actual_values[:, 0]])
            print('pred_Teff =', [f'{x:.6f}' for x in all_predictions[:, 0]])
            print('actual_logg =', [f'{x:.6f}' for x in all_actual_values[:, 1]])
            print('pred_logg =', [f'{x:.6f}' for x in all_predictions[:, 1]])
            print('actual_logL =', [f'{x:.6f}' for x in actual_logL])
            print('pred_logL =', [f'{x:.6f}' for x in pred_logL])
            print(f'{"="*60}\n')
    
    elif predict_target == 'teff_logl':
        # For MLP2 - directly predicting logL
        if True: # Always plot for final evaluation
            plt.figure(figsize=(10, 6))
            plt.scatter(all_actual_values[:, 1], all_predictions[:, 1], alpha=0.3)
            plt.xlabel('Actual logL')
            plt.ylabel('Predicted logL')
            plt.title('Predicted vs Actual logL (Direct)')
            plt.plot([all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], [all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logL_direct_final.png")
            wandb.log({"MLP Predicted vs Actual logL (Direct) Final": wandb.Image("pred_vs_act_logL_direct_final.png", caption="Predictions vs. Actual logL (Direct) at Epoch {}".format(epoch_name))})
            plt.close()

        # Calculate and print metrics
        r2_teff = r2_score(all_actual_values[:, 0], all_predictions[:, 0])
        r2_logl = r2_score(all_actual_values[:, 1], all_predictions[:, 1])
        mse_teff = mean_squared_error(all_actual_values[:, 0], all_predictions[:, 0])
        mse_logl = mean_squared_error(all_actual_values[:, 1], all_predictions[:, 1])
        
        print(f'R2 score of Teff at Epoch {epoch_name}: {r2_teff:.6f}')
        print(f'R2 score of logL (direct) at Epoch {epoch_name}: {r2_logl:.6f}')
        print(f'MSE of Teff at Epoch {epoch_name}: {mse_teff:.6f}')
        print(f'MSE of logL (direct) at Epoch {epoch_name}: {mse_logl:.6f}')
        
        # Output detailed arrays only if requested
        if True: # Always print for final evaluation
            print(f'\n{"="*60}')
            print(f'DETAILED ARRAYS FOR PLOTTING (Epoch {epoch_name})')
            print(f'{"="*60}')
            print('actual_Teff =', [f'{x:.6f}' for x in all_actual_values[:, 0]])
            print('pred_Teff =', [f'{x:.6f}' for x in all_predictions[:, 0]])
            print('actual_logL =', [f'{x:.6f}' for x in all_actual_values[:, 1]])
            print('pred_logL =', [f'{x:.6f}' for x in all_predictions[:, 1]])
            print(f'{"="*60}\n')

    return all_actual_values, all_predictions

def run_experiment(exp_name, predict_target, hidden_sizes, seed, epochs=10000, patience=300):
    """Run a single experiment with specified parameters - EXACTLY like ablation study"""
    
    # Set the random seed EXACTLY like the ablation study
    set_all_seeds(seed)
    
    # Initialize wandb
    wandb.init(project="lightcurve-to-spectra-ml-benchmark-mlp-reg", 
               entity="rczhang",
               name=exp_name,
               config={
                   "experiment": exp_name,
                   "predict_target": predict_target,
                   "hidden_sizes": hidden_sizes,
                   "seed": seed,
                   "epochs": epochs,
                   "patience": patience,
                   "batch_size": 32
               })
    
    # Load data
    alpha0, nu_char, gamma, Cw, Teff, logg, Msp = load_data('curvefitparams_reg.h5')
    batch_size = 32
    data_normalized, labels_tensor, normalization_params = preprocess_data(alpha0, nu_char, gamma, Cw, Teff, logg, Msp, predict_target)
    print('data normalized', data_normalized)
    
    # Create dataloaders with the SAME seed as the ablation study
    train_loader, test_loader, test_dataset = create_dataloaders(data_normalized, labels_tensor, batch_size, seed)
    loss_fn = nn.MSELoss().cuda()
    
    # Create model with specified architecture
    model = MLP(input_size=4, hidden_sizes=hidden_sizes, output_size=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    # Use the EXACT same train_model function as ablation study
    def train_model_exact(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            
            # Training - NO EXTERNAL FUNCTION CALLS
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            train_loss = total_loss / len(train_loader)
            print(f"Train loss: {train_loss:>7f}")
            
            # Validation - NO EXTERNAL FUNCTION CALLS
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.cuda(), y.cuda()
                    pred = model(X)
                    val_loss += loss_fn(pred, y).item()
            
            val_loss /= len(test_loader)
            print(f"Test loss: {val_loss:>7f}")
            
            # Log to wandb
            wandb.log({"train_loss": train_loss, "test_loss": val_loss, "epoch": epoch})
            
            # Early stopping logic
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"New best model saved! Loss: {best_loss:>7f}")
            else:
                patience_counter += 1
                
            print(f"Patience counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # Load the best model state before returning
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with loss: {best_loss:>7f}")
        
        return epoch + 1, best_loss
    
    # Use the EXACT same evaluate_model function as ablation study
    def evaluate_model_exact(model, test_loader, norm_params, predict_target):
        model.eval()
        all_preds = []
        all_ys = []
        
        Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                all_preds.extend(pred.cpu().numpy())
                all_ys.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_ys = np.array(all_ys)
        
        # Denormalize based on prediction target
        all_preds_denorm = all_preds.copy()
        all_ys_denorm = all_ys.copy()
        
        if predict_target == 'teff_logg':
            # Predicting Teff and logg
            all_preds_denorm[:, 0] = all_preds[:, 0] * Teff_std + Teff_mean
            all_preds_denorm[:, 1] = all_preds[:, 1] * logg_std + logg_mean
            all_ys_denorm[:, 0] = all_ys[:, 0] * Teff_std + Teff_mean
            all_ys_denorm[:, 1] = all_ys[:, 1] * logg_std + logg_mean
            
            # Calculate derived logL
            actual_logL = calculate_lum_from_teff_logg(all_ys_denorm[:, 0], all_ys_denorm[:, 1], False)
            pred_logL = calculate_lum_from_teff_logg(all_preds_denorm[:, 0], all_preds_denorm[:, 1], False)
            
            # Calculate metrics
            r2_teff = r2_score(all_ys_denorm[:, 0], all_preds_denorm[:, 0])
            r2_logg = r2_score(all_ys_denorm[:, 1], all_preds_denorm[:, 1])
            r2_logl = r2_score(actual_logL, pred_logL)
            
            rmse_teff = np.sqrt(mean_squared_error(all_ys_denorm[:, 0], all_preds_denorm[:, 0]))
            rmse_logg = np.sqrt(mean_squared_error(all_ys_denorm[:, 1], all_preds_denorm[:, 1]))
            rmse_logl = np.sqrt(mean_squared_error(actual_logL, pred_logL))
            
        elif predict_target == 'teff_logl':
            # Predicting Teff and logL directly
            all_preds_denorm[:, 0] = all_preds[:, 0] * Teff_std + Teff_mean
            all_preds_denorm[:, 1] = all_preds[:, 1] * logL_std + logL_mean
            all_ys_denorm[:, 0] = all_ys[:, 0] * Teff_std + Teff_mean
            all_ys_denorm[:, 1] = all_ys[:, 1] * logL_std + logL_mean
            
            # Calculate metrics
            r2_teff = r2_score(all_ys_denorm[:, 0], all_preds_denorm[:, 0])
            r2_logl = r2_score(all_ys_denorm[:, 1], all_preds_denorm[:, 1])
            
            rmse_teff = np.sqrt(mean_squared_error(all_ys_denorm[:, 0], all_preds_denorm[:, 0]))
            rmse_logl = np.sqrt(mean_squared_error(all_ys_denorm[:, 1], all_preds_denorm[:, 1]))
            
            # Placeholder for logg
            r2_logg = 0.0
            rmse_logg = 0.0
        
        return r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl, all_ys_denorm, all_preds_denorm
    
    # Train model EXACTLY like ablation study
    epochs_trained, final_loss = train_model_exact(model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs, patience, normalization_params)
    
    # Evaluate model EXACTLY like ablation study
    r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl, all_ys_denorm, all_preds_denorm = evaluate_model_exact(model, test_loader, normalization_params, predict_target)
    
    # Print results exactly like ablation study
    print(f'R2 score of Teff: {r2_teff:.6f}')
    if predict_target == 'teff_logg':
        print(f'R2 score of logg: {r2_logg:.6f}')
    print(f'R2 score of logL: {r2_logl:.6f}')
    print(f'Final Loss: {final_loss:.6f}')
    print(f'Epochs: {epochs_trained}')
    
    # Create final plots with results
    final_evaluation_with_stored_results(all_ys_denorm, all_preds_denorm, normalization_params, predict_target, "FINAL")
    
    print("Done!")
    wandb.finish()

if __name__ == '__main__':
    # Experiment configurations based on ablation study results
    experiments = [
        {
            'name': 'MLP1_median_3layer-256-512-128',
            'predict_target': 'teff_logg',
            'hidden_sizes': [256, 512, 128],
            'seed': 192021
        },
        {
            'name': 'MLP2_median_3layer-512-256-128', 
            'predict_target': 'teff_logl',
            'hidden_sizes': [512, 256, 128],
            'seed': 456
        }
    ]
    
    # Run both experiments
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp['name']}")
        print(f"Architecture: {exp['hidden_sizes']}")
        print(f"Target: {exp['predict_target']}")
        print(f"Seed: {exp['seed']}")
        print(f"{'='*60}\n")
        
        run_experiment(
            exp_name=exp['name'],
            predict_target=exp['predict_target'],
            hidden_sizes=exp['hidden_sizes'],
            seed=exp['seed'],
            epochs=500,
            patience=50
        )
