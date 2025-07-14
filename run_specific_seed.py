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
from data import read_hdf5_data
from run_mlp import apply_min_max_scaling

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

# MLP Model
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

# CNN Model
class CNN1D(nn.Module):
    def __init__(self, num_channels, fc_size, output_size, input_size, num_conv_layers=3):
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
        torch.manual_seed(12345)  # Fixed seed for dummy tensor
        dummy_input = torch.randn(1, 1, input_size)
        dummy_output = self.conv_layers(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]
        
        self.flatten = nn.Flatten()
        
        # Build fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.output_dim, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, output_size)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        output = self.fc_layers(x)
        return output

def load_data_mlp(filename):
    """Load data for MLP from curvefitparams_reg.h5"""
    data = pd.read_hdf(filename)
    alpha0 = data['alpha0']
    nu_char = data['nu_char']
    gamma = data['gamma']
    Cw = data['Cw']
    Teff = data['Teff']
    logg = data['logg']
    Msp = data['Msp']
    return alpha0, nu_char, gamma, Cw, Teff, logg, Msp

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

def preprocess_data_mlp(alpha0, nu_char, gamma, Cw, Teff, logg, Msp, predict_target='teff_logl'):
    """Preprocess data for MLP"""
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
    if predict_target == 'teff_logl':
        labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)
    else:
        raise ValueError("predict_target must be 'teff_logl' for MLP2")

    return data_normalized, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std)

def preprocess_data_cnn(power, Teff, logg, Msp, predict_target='teff_logg'):
    """Preprocess data for CNN"""
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
    else:
        raise ValueError("predict_target must be 'teff_logg' for CNN1")

    return power_tensor, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std)

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

def denormalize_data(norm_data, means, stds):
    denorm_data = []
    for data in norm_data:
        denorm_sublist = [(value * std + mean) for value, mean, std in zip(data, means, stds)]
        denorm_data.append(denorm_sublist)
    return np.array(denorm_data)

def run_mlp2_experiment(seed):
    """Run MLP2 experiment with specified seed"""
    print(f"\n{'='*60}")
    print(f"Running MLP2 experiment with seed {seed}")
    print(f"{'='*60}\n")
    
    # Set the random seed
    set_all_seeds(seed)
    
    # Initialize wandb
    wandb.init(project="lightcurve-to-spectra-ml-benchmark-specific-seed", 
               entity="rczhang",
               name=f"MLP2_seed_{seed}",
               config={
                   "experiment": f"MLP2_seed_{seed}",
                   "predict_target": "teff_logl",
                   "hidden_sizes": [512, 256, 128],
                   "seed": seed,
                   "epochs": 500,
                   "patience": 50,
                   "batch_size": 32
               })
    
    # Load data
    alpha0, nu_char, gamma, Cw, Teff, logg, Msp = load_data_mlp('curvefitparams_reg.h5')
    batch_size = 32
    data_normalized, labels_tensor, normalization_params = preprocess_data_mlp(alpha0, nu_char, gamma, Cw, Teff, logg, Msp, 'teff_logl')
    
    # Create dataloaders
    train_loader, test_loader, test_dataset = create_dataloaders(data_normalized, labels_tensor, batch_size, seed)
    loss_fn = nn.MSELoss().cuda()
    
    # Create model
    model = MLP(input_size=4, hidden_sizes=[512, 256, 128], output_size=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    # Training function
    def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            
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
            
            train_loss = total_loss / len(train_loader)
            print(f"Train loss: {train_loss:>7f}")
            
            # Validation
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
        
        # Load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with loss: {best_loss:>7f}")
        
        return epoch + 1, best_loss
    
    # Evaluation function
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
        
        # Calculate metrics
        r2_teff = r2_score(all_ys_denorm[:, 0], all_preds_denorm[:, 0])
        r2_logl = r2_score(all_ys_denorm[:, 1], all_preds_denorm[:, 1])
        
        return r2_teff, r2_logl, all_ys_denorm, all_preds_denorm
    
    # Train model
    epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, 500, 50, normalization_params)
    
    # Evaluate model
    r2_teff, r2_logl, all_ys_denorm, all_preds_denorm = evaluate_model(model, test_loader, normalization_params)
    
    # Print results
    print(f'\nMLP2 Results:')
    print(f'R2 score of Teff: {r2_teff:.6f}')
    print(f'R2 score of logL: {r2_logl:.6f}')
    print(f'Final Loss: {final_loss:.6f}')
    print(f'Epochs: {epochs_trained}')
    
    # Print detailed arrays for plotting
    print(f'\n{"="*60}')
    print(f'MLP2 DETAILED ARRAYS FOR PLOTTING (Seed {seed})')
    print(f'{"="*60}')
    print('actual_Teff =', [f'{x:.6f}' for x in all_ys_denorm[:, 0]])
    print('pred_Teff =', [f'{x:.6f}' for x in all_preds_denorm[:, 0]])
    print('actual_logL =', [f'{x:.6f}' for x in all_ys_denorm[:, 1]])
    print('pred_logL =', [f'{x:.6f}' for x in all_preds_denorm[:, 1]])
    print(f'{"="*60}\n')
    
    wandb.finish()
    return r2_teff, r2_logl

def run_cnn1_experiment(seed):
    """Run CNN1 experiment with specified seed"""
    print(f"\n{'='*60}")
    print(f"Running CNN1 experiment with seed {seed}")
    print(f"{'='*60}\n")
    
    # Set the random seed
    set_all_seeds(seed)
    
    # Initialize wandb
    wandb.init(project="lightcurve-to-spectra-ml-benchmark-specific-seed", 
               entity="rczhang",
               name=f"CNN1_seed_{seed}",
               config={
                   "experiment": f"CNN1_seed_{seed}",
                   "predict_target": "teff_logg",
                   "architecture": "3conv-64ch-128fc",
                   "seed": seed,
                   "epochs": 500,
                   "patience": 50,
                   "batch_size": 32
               })
    
    # Load data
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data_cnn(power, Teff, logg, Msp, 'teff_logg')
    
    batch_size = 32
    input_size = len(power.iloc[0])
    
    # Create dataloaders
    train_loader, test_loader, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size, seed)
    loss_fn = nn.MSELoss().cuda()
    
    # Create model
    model = CNN1D(num_channels=64, fc_size=128, output_size=2, input_size=input_size, num_conv_layers=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    
    # Training function
    def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            
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
            
            train_loss = total_loss / len(train_loader)
            print(f"Train loss: {train_loss:>7f}")
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.cuda().unsqueeze(1), y.cuda()
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
        
        # Load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with loss: {best_loss:>7f}")
        
        return epoch + 1, best_loss
    
    # Evaluation function
    def evaluate_model(model, test_loader, norm_params):
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
        
        # Denormalize
        all_preds_denorm = all_preds.copy()
        all_ys_denorm = all_ys.copy()
        
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
        
        return r2_teff, r2_logg, r2_logl, all_ys_denorm, all_preds_denorm, actual_logL, pred_logL
    
    # Train model
    epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, 500, 50, normalization_params)
    
    # Evaluate model
    r2_teff, r2_logg, r2_logl, all_ys_denorm, all_preds_denorm, actual_logL, pred_logL = evaluate_model(model, test_loader, normalization_params)
    
    # Print results
    print(f'\nCNN1 Results:')
    print(f'R2 score of Teff: {r2_teff:.6f}')
    print(f'R2 score of logg: {r2_logg:.6f}')
    print(f'R2 score of logL: {r2_logl:.6f}')
    print(f'Final Loss: {final_loss:.6f}')
    print(f'Epochs: {epochs_trained}')
    
    # Print detailed arrays for plotting
    print(f'\n{"="*60}')
    print(f'CNN1 DETAILED ARRAYS FOR PLOTTING (Seed {seed})')
    print(f'{"="*60}')
    print('actual_Teff =', [f'{x:.6f}' for x in all_ys_denorm[:, 0]])
    print('pred_Teff =', [f'{x:.6f}' for x in all_preds_denorm[:, 0]])
    print('actual_logL =', [f'{x:.6f}' for x in actual_logL])
    print('pred_logL =', [f'{x:.6f}' for x in pred_logL])
    print(f'{"="*60}\n')
    
    wandb.finish()
    return r2_teff, r2_logl

if __name__ == '__main__':
    # Run both experiments with seed 222324
    seed = 222324
    
    # Run MLP2 experiment
    mlp2_r2_teff, mlp2_r2_logl = run_mlp2_experiment(seed)
    
    # Run CNN1 experiment  
    cnn1_r2_teff, cnn1_r2_logl = run_cnn1_experiment(seed)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY FOR SEED {seed}")
    print(f"{'='*80}")
    print(f"MLP2 Results:")
    print(f"  R² Teff: {mlp2_r2_teff:.4f}")
    print(f"  R² logL: {mlp2_r2_logl:.4f}")
    print(f"CNN1 Results:")
    print(f"  R² Teff: {cnn1_r2_teff:.4f}")
    print(f"  R² logL: {cnn1_r2_logl:.4f}")
    print(f"{'='*80}")
    
    print("Done!") 