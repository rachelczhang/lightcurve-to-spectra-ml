import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv

torch.manual_seed(42)
np.random.seed(42)

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def calculate_lum_from_teff_logg(Teff, logg, tensor):
    if tensor == False:
        log_L_sun=np.log10((5777)**4.0/(274*100))  # Used for Spectroscopic HRD
        Teff_K = Teff * 1000
        logL_solar = 4*np.log10(Teff_K)-logg-log_L_sun
    else:
        log_L_sun=torch.tensor((np.log10((5777)**4.0/(274*100))), dtype=torch.float64)
        Teff_K = Teff * 1000
        logL_solar = 4*torch.log10(Teff_K)-logg-log_L_sun
    return logL_solar

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
    Teff_norm, Teff_mean, Teff_std = normalize_data(np.array(Teff))
    logg_norm, logg_mean, logg_std = normalize_data(np.array(logg))
    Msp_norm, Msp_mean, Msp_std = normalize_data(np.array(Msp))
    logL = calculate_lum_from_teff_logg(np.array(Teff), np.array(logg), False)
    logL_norm, logL_mean, logL_std = normalize_data(logL)

    labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)

    return data_normalized, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std)

def create_dataloaders(power_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return epoch + 1, best_loss

def evaluate_model(model, test_loader, norm_params):
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
    
    # Denormalize
    all_preds_denorm = all_preds.copy()
    all_ys_denorm = all_ys.copy()
    all_preds_denorm[:, 0] = all_preds[:, 0] * Teff_std + Teff_mean
    all_preds_denorm[:, 1] = all_preds[:, 1] * logL_std + logL_mean
    all_ys_denorm[:, 0] = all_ys[:, 0] * Teff_std + Teff_mean
    all_ys_denorm[:, 1] = all_ys[:, 1] * logL_std + logL_mean
    
    # # Calculate luminosities
    # actual_logL = calculate_lum_from_teff_logg(all_ys_denorm[:, 0], all_ys_denorm[:, 1], False)
    # pred_logL = calculate_lum_from_teff_logg(all_preds_denorm[:, 0], all_preds_denorm[:, 1], False)
    
    # Calculate metrics
    r2_teff = r2_score(all_ys_denorm[:, 0], all_preds_denorm[:, 0])
    r2_logl = r2_score(all_ys_denorm[:, 1], all_preds_denorm[:, 1])
    # r2_logg = r2_score(all_ys_denorm[:, 1], all_preds_denorm[:, 1])
    # r2_logl = r2_score(actual_logL, pred_logL)
    
    rmse_teff = np.sqrt(mean_squared_error(all_ys_denorm[:, 0], all_preds_denorm[:, 0]))
    rmse_logl = np.sqrt(mean_squared_error(all_ys_denorm[:, 1], all_preds_denorm[:, 1]))
    # rmse_logg = np.sqrt(mean_squared_error(all_ys_denorm[:, 1], all_preds_denorm[:, 1]))
    # rmse_logl = np.sqrt(mean_squared_error(actual_logL, pred_logL))
    
    # Return dummy values for logg since we're not predicting it
    r2_logg = 0.0  # Placeholder
    rmse_logg = 0.0  # Placeholder
    
    return r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl

def run_ablation_study():
    # Load data once
    alpha0, nu_char, gamma, Cw, Teff, logg, Msp = load_data('curvefitparams_reg.h5')
    data_normalized, labels_tensor, normalization_params = preprocess_data(alpha0, nu_char, gamma, Cw, Teff, logg, Msp)
    
    # Define architectures to test
    architectures = [
        # 1 hidden layer 
        ([128]),
        ([256]),
        ([512]),
        
        # 2 hidden layers - representative patterns
        ([256, 128]),
        ([128, 256]),
        ([256, 256]),
        
        # 3 hidden layers 
        ([256, 128, 64]),
        ([512, 256, 128]), 
        ([128, 256, 128]),  
        ([256, 512, 128]),
        ([256, 256, 256])
    ]
    
    results = []
    batch_size = 32
    max_epochs = 1000 
    patience = 100
    
    print("Running ablation study...")
    print("Architecture | Params | Epochs | Train Time | R²(Teff) | R²(logg) | R²(logL) | RMSE(Teff) | RMSE(logg) | RMSE(logL)")
    print("-" * 120)
    
    for i, hidden_sizes in enumerate(architectures):
        start_time = time.time()
        
        # Create data loaders (new split each time for fair comparison)
        train_loader, test_loader, _ = create_dataloaders(data_normalized, labels_tensor, batch_size)
        
        # Create model
        model = MLP(input_size=4, hidden_sizes=hidden_sizes, output_size=2).cuda()
        num_params = count_parameters(model)
        
        # Training setup
        loss_fn = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
        
        # Train model
        epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, normalization_params)
        
        # Evaluate model
        r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl = evaluate_model(model, test_loader, normalization_params)
        
        train_time = time.time() - start_time
        
        # Store results
        arch_str = str(hidden_sizes)
        result = {
            'Architecture': arch_str,
            'Parameters': num_params,
            'Epochs': epochs_trained,
            'Train_Time_min': train_time / 60,
            'R2_Teff': r2_teff,
            'R2_logg': r2_logg,
            'R2_logL': r2_logl,
            'RMSE_Teff': rmse_teff,
            'RMSE_logg': rmse_logg,
            'RMSE_logL': rmse_logl,
            'Final_Loss': final_loss
        }
        results.append(result)
        
        # Print progress
        print(f"{arch_str:15} | {num_params:6} | {epochs_trained:6} | {train_time/60:8.1f} | {r2_teff:8.3f} | {r2_logg:8.3f} | {r2_logl:8.3f} | {rmse_teff:10.1f} | {rmse_logg:10.3f} | {rmse_logl:10.3f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('ablation_study_mlp2_results.csv', index=False)
    print(f"\nResults saved to ablation_study_mlp2_results.csv")
    
    # Find best models
    best_r2_logl = df.loc[df['R2_logL'].idxmax()]
    best_rmse_logl = df.loc[df['RMSE_logL'].idxmin()]
    
    print(f"\nBest R² (logL): {best_r2_logl['Architecture']} with R² = {best_r2_logl['R2_logL']:.3f}")
    print(f"Best RMSE (logL): {best_rmse_logl['Architecture']} with RMSE = {best_rmse_logl['RMSE_logL']:.3f}")
    
    return results

if __name__ == '__main__':
    results = run_ablation_study() 