import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import read_hdf5_data
from run_mlp import apply_min_max_scaling

torch.manual_seed(42)
np.random.seed(42)

class CNN1D(nn.Module):
    def __init__(self, num_channels, fc_size, output_size, input_size, num_conv_layers=2):
        """
        num_channels: # of output channels for convolutional layers
        fc_size: size of the fully connected hidden layer
        output_size: # of outputs (2 for Teff+logg or Teff+logL)
        input_size: length of input power spectrum
        num_conv_layers: number of convolutional layers (1, 2, or 3)
        """
        super().__init__()
        
        conv_layers = []
        in_channels = 1
        current_size = input_size
        
        # Build convolutional layers
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, num_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ])
            in_channels = num_channels
            current_size = current_size // 2
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output dimension after conv layers
        dummy_input = torch.randn(1, 1, input_size)
        dummy_output = self.conv_layers(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]
        
        self.flatten = nn.Flatten()
        
        # Build fully connected layers
        if isinstance(fc_size, tuple):
            # Two FC layers
            fc1_size, fc2_size = fc_size
            self.fc_layers = nn.Sequential(
                nn.Linear(self.output_dim, fc1_size),
                nn.ReLU(),
                nn.Linear(fc1_size, fc2_size),
                nn.ReLU(),
                nn.Linear(fc2_size, output_size)
            )
        elif fc_size > 0:
            # Single FC layer
            self.fc_layers = nn.Sequential(
                nn.Linear(self.output_dim, fc_size),
                nn.ReLU(),
                nn.Linear(fc_size, output_size)
            )
        else:
            # Direct connection from conv to output
            self.fc_layers = nn.Linear(self.output_dim, output_size)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        output = self.fc_layers(x)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_lum_from_teff_logg(Teff, logg, tensor):
    if tensor == False:
        log_L_sun = np.log10((5777)**4.0/(274*100))
        Teff_K = Teff * 1000
        logL_solar = 4*np.log10(Teff_K) - logg - log_L_sun
    else:
        log_L_sun = torch.tensor((np.log10((5777)**4.0/(274*100))), dtype=torch.float64)
        Teff_K = Teff * 1000
        logL_solar = 4*torch.log10(Teff_K) - logg - log_L_sun
    return logL_solar

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

def preprocess_data(power, Teff, logg, Msp, predict_target='teff_logg'):
    """
    predict_target: 'teff_logg' or 'teff_logl'
    """
    Teff = [float(t) for t in Teff]
    logg = [float(l) for l in logg]
    Msp = [float(m) for m in Msp]
    
    Teff_norm, Teff_mean, Teff_std = normalize_data(np.array(Teff))
    logg_norm, logg_mean, logg_std = normalize_data(np.array(logg))
    Msp_norm, Msp_mean, Msp_std = normalize_data(np.array(Msp))
    logL = calculate_lum_from_teff_logg(np.array(Teff), np.array(logg), False)
    logL_norm, logL_mean, logL_std = normalize_data(logL)

    # Log-transform and scale the power spectrum
    logpower = power.apply(lambda x: np.log10(x).tolist())
    scaled_power = apply_min_max_scaling(logpower)
    power_tensor = torch.tensor(np.array(scaled_power.tolist(), dtype=np.float32))
    
    # Create labels based on prediction target
    if predict_target == 'teff_logg':
        labels_tensor = torch.tensor(list(zip(Teff_norm, logg_norm)), dtype=torch.float32)
    elif predict_target == 'teff_logl':
        labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)
    else:
        raise ValueError("predict_target must be 'teff_logg' or 'teff_logl'")

    return power_tensor, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std)

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
            X, y = X.cuda().unsqueeze(1), y.cuda()  # Add channel dimension for CNN
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
                X, y = X.cuda().unsqueeze(1), y.cuda()
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return epoch + 1, best_loss

def evaluate_model(model, test_loader, norm_params, predict_target):
    model.eval()
    all_preds = []
    all_ys = []
    
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.cuda().unsqueeze(1), y.cuda()
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
    
    return r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl

def run_ablation_study(predict_target='teff_logg'):
    """
    predict_target: 'teff_logg' or 'teff_logl'
    """
    # Load data
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    input_size = len(power.iloc[0])
    
    # Define CNN architectures to test
    # Format: (num_channels, fc_size, num_conv_layers, description)
    architectures = [
        # 1 Conv layer variations
        (16, 64, 1, "1conv-16ch-64fc"),
        (32, 64, 1, "1conv-32ch-64fc"),
        (32, 128, 1, "1conv-32ch-128fc"),
        (64, 128, 1, "1conv-64ch-128fc"),
        
        # 2 Conv layer variations 
        (16, 64, 2, "2conv-16ch-64fc"),
        (32, 64, 2, "2conv-32ch-64fc"),
        (32, 128, 2, "2conv-32ch-128fc"),  
        (64, 128, 2, "2conv-64ch-128fc"),
        (32, 256, 2, "2conv-32ch-256fc"),
        
        # 3 Conv layer variations
        (16, 64, 3, "3conv-16ch-64fc"),
        (32, 64, 3, "3conv-32ch-64fc"),
        (32, 128, 3, "3conv-32ch-128fc"),
        (64, 128, 3, "3conv-64ch-128fc"),
        
        # 2 FC layer variations (deeper FC networks)
        (32, (128, 64), 2, "2conv-32ch-128fc-64fc"),
        (32, (256, 128), 2, "2conv-32ch-256fc-128fc"),
        (64, (128, 64), 2, "2conv-64ch-128fc-64fc"),
    ]
    
    results = []
    batch_size = 32
    max_epochs = 500  # Reduced for efficiency
    patience = 50
    
    target_name = "Teff+logg" if predict_target == 'teff_logg' else "Teff+logL"
    print(f"Running CNN ablation study for {target_name} prediction...")
    print("Architecture | Params | Epochs | Train Time | R²(Teff) | R²(logg) | R²(logL) | RMSE(Teff) | RMSE(logg) | RMSE(logL)")
    print("-" * 130)
    
    for i, (num_channels, fc_size, num_conv_layers, arch_desc) in enumerate(architectures):
        start_time = time.time()
        
        # Create data loaders (new split each time for fair comparison)
        train_loader, test_loader, _ = create_dataloaders(power_tensor, labels_tensor, batch_size)
        
        # Create model
        model = CNN1D(num_channels, fc_size, 2, input_size, num_conv_layers).cuda()
        num_params = count_parameters(model)
        
        # Training setup
        loss_fn = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
        
        # Train model
        epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, normalization_params)
        
        # Evaluate model
        r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl = evaluate_model(model, test_loader, normalization_params, predict_target)
        
        train_time = time.time() - start_time
        
        # Store results
        result = {
            'Architecture': arch_desc,
            'Channels': num_channels,
            'FC_Size': fc_size,
            'Conv_Layers': num_conv_layers,
            'Parameters': num_params,
            'Epochs': epochs_trained,
            'Train_Time_min': train_time / 60,
            'R2_Teff': r2_teff,
            'R2_logg': r2_logg,
            'R2_logL': r2_logl,
            'RMSE_Teff': rmse_teff,
            'RMSE_logg': rmse_logg,
            'RMSE_logL': rmse_logl,
            'Final_Loss': final_loss,
            'Predict_Target': predict_target
        }
        results.append(result)
        
        # Print progress
        print(f"{arch_desc:15} | {num_params:6} | {epochs_trained:6} | {train_time/60:8.1f} | {r2_teff:8.3f} | {r2_logg:8.3f} | {r2_logl:8.3f} | {rmse_teff:10.1f} | {rmse_logg:10.3f} | {rmse_logl:10.3f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    filename = f'ablation_study_cnn_{predict_target}_results.csv'
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Find best models
    best_r2_logl = df.loc[df['R2_logL'].idxmax()]
    best_rmse_logl = df.loc[df['RMSE_logL'].idxmin()]
    
    print(f"\nBest R² (logL): {best_r2_logl['Architecture']} with R² = {best_r2_logl['R2_logL']:.3f}")
    print(f"Best RMSE (logL): {best_rmse_logl['Architecture']} with RMSE = {best_rmse_logl['RMSE_logL']:.3f}")
    
    return results

if __name__ == '__main__':
    # Run ablation for both prediction targets
    print("=" * 50)
    print("ABLATION STUDY 1: Predicting Teff + logg")
    print("=" * 50)
    results_teff_logg = run_ablation_study('teff_logg')
    
    print("\n" + "=" * 50)
    print("ABLATION STUDY 2: Predicting Teff + logL")
    print("=" * 50)
    results_teff_logl = run_ablation_study('teff_logl') 