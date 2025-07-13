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

class CNN1D(nn.Module):
    def __init__(self, num_channels, fc_size, output_size, input_size, num_conv_layers=3):
        """
        CNN with 3 conv layers, 64 channels, 128 FC layer
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
        X = X.cuda().unsqueeze(1)  # Add channel dimension for CNN
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
            X, y = X.cuda().unsqueeze(1), y.cuda()  # Add channel dimension for CNN
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
        wandb.log({"CNN Predicted vs Actual Teff": wandb.Image("pred_vs_act_Teff.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch))})
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
            wandb.log({"CNN Predicted vs Actual logg": wandb.Image("pred_vs_act_logg.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch))})
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
        # For CNN2 - directly predicting logL
        if output_details:
            plt.figure(figsize=(10, 6))
            plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
            plt.xlabel('Actual logL')
            plt.ylabel('Predicted logL')
            plt.title('Predicted vs Actual logL (Direct)')
            plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
            plt.grid(True)
            plt.savefig("pred_vs_act_logL_direct.png")
            wandb.log({"CNN Predicted vs Actual logL (Direct)": wandb.Image("pred_vs_act_logL_direct.png", caption="Predictions vs. Actual logL (Direct) at Epoch {}".format(epoch))})
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
    """
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
   
    # Always create Teff plot for wandb
    plt.figure(figsize=(10, 6))
    plt.scatter(all_actual_values[:, 0], all_predictions[:, 0], alpha=0.3)
    plt.xlabel('Actual Teff')
    plt.ylabel('Predicted Teff')
    plt.title('Predicted vs Actual Teff')
    plt.plot([all_actual_values[:, 0].min(), all_actual_values[:, 0].max()], [all_actual_values[:, 0].min(), all_actual_values[:, 0].max()], 'r')
    plt.grid(True)
    plt.savefig("pred_vs_act_Teff_final.png")
    wandb.log({"CNN Predicted vs Actual Teff Final": wandb.Image("pred_vs_act_Teff_final.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch_name))})
    plt.close()

    if predict_target == 'teff_logg':
        # Calculate derived logL
        actual_logL = calculate_lum_from_teff_logg(np.array(all_actual_values[:, 0], dtype=np.float64), np.array(all_actual_values[:, 1], dtype=np.float64), False)
        pred_logL = calculate_lum_from_teff_logg(np.array(all_predictions[:, 0], dtype=np.float64), np.array(all_predictions[:, 1], dtype=np.float64), False)
        
        # Create plots
        plt.figure(figsize=(10, 6))
        plt.scatter(all_actual_values[:, 1], all_predictions[:, 1], alpha=0.3)
        plt.xlabel('Actual logg')
        plt.ylabel('Predicted logg')
        plt.title('Predicted vs Actual logg')
        plt.plot([all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], [all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], 'r')
        plt.grid(True)
        plt.savefig("pred_vs_act_logg_final.png")
        wandb.log({"CNN Predicted vs Actual logg Final": wandb.Image("pred_vs_act_logg_final.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch_name))})
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(actual_logL, pred_logL, alpha=0.3)
        plt.xlabel('Actual logL')
        plt.ylabel('Predicted logL')
        plt.title('Predicted vs Actual Spectroscopic logL')
        plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
        plt.grid(True)
        plt.savefig("pred_vs_act_logL_final.png")
        wandb.log({"CNN Predicted vs Actual logL Final": wandb.Image("pred_vs_act_logL_final.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch_name))})
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
        
        # Output detailed arrays
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
        # For CNN2 - directly predicting logL
        plt.figure(figsize=(10, 6))
        plt.scatter(all_actual_values[:, 1], all_predictions[:, 1], alpha=0.3)
        plt.xlabel('Actual logL')
        plt.ylabel('Predicted logL')
        plt.title('Predicted vs Actual logL (Direct)')
        plt.plot([all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], [all_actual_values[:, 1].min(), all_actual_values[:, 1].max()], 'r')
        plt.grid(True)
        plt.savefig("pred_vs_act_logL_direct_final.png")
        wandb.log({"CNN Predicted vs Actual logL (Direct) Final": wandb.Image("pred_vs_act_logL_direct_final.png", caption="Predictions vs. Actual logL (Direct) at Epoch {}".format(epoch_name))})
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
        
        # Output detailed arrays
        print(f'\n{"="*60}')
        print(f'DETAILED ARRAYS FOR PLOTTING (Epoch {epoch_name})')
        print(f'{"="*60}')
        print('actual_Teff =', [f'{x:.6f}' for x in all_actual_values[:, 0]])
        print('pred_Teff =', [f'{x:.6f}' for x in all_predictions[:, 0]])
        print('actual_logL =', [f'{x:.6f}' for x in all_actual_values[:, 1]])
        print('pred_logL =', [f'{x:.6f}' for x in all_predictions[:, 1]])
        print(f'{"="*60}\n')

    return all_actual_values, all_predictions

def run_experiment(exp_name, predict_target, seed, epochs=500, patience=50):
    """Run a single experiment with specified parameters - EXACTLY like ablation study"""
    
    # Set the random seed EXACTLY like the ablation study
    set_all_seeds(seed)
    
    # Initialize wandb
    wandb.init(project="lightcurve-to-spectra-ml-regression", 
               entity="rczhang",
               name=exp_name,
               config={
                   "experiment": exp_name,
                   "predict_target": predict_target,
                   "architecture": "3conv-64ch-128fc",
                   "seed": seed,
                   "epochs": epochs,
                   "patience": patience,
                   "batch_size": 32
               })
    
    # Load data
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    batch_size = 32
    input_size = len(power.iloc[0])
    
    # Create dataloaders with the SAME seed as the ablation study
    train_loader, test_loader, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size, seed)
    loss_fn = nn.MSELoss().cuda()
    
    # Create model with 3conv-64ch-128fc architecture EXACTLY like ablation study
    model = CNN1D(num_channels=64, fc_size=128, output_size=2, input_size=input_size, num_conv_layers=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    
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
                X, y = X.cuda().unsqueeze(1), y.cuda()  # Add channel dimension for CNN
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
    # Experiment configurations based on ablation study median results
    experiments = [
        {
            'name': 'CNN1_median_3conv-64ch-128fc',
            'predict_target': 'teff_logg',
            'seed': 131415  # Median seed for CNN1 
        },
        {
            'name': 'CNN2_median_3conv-64ch-128fc', 
            'predict_target': 'teff_logl',
            'seed': 363738  # Median seed for CNN2
        }
    ]
    
    # Run both experiments
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp['name']}")
        print(f"Architecture: 3conv-64ch-128fc")
        print(f"Target: {exp['predict_target']}")
        print(f"Seed: {exp['seed']}")
        print(f"{'='*60}\n")
        
        run_experiment(
            exp_name=exp['name'],
            predict_target=exp['predict_target'],
            seed=exp['seed'],
            epochs=500,
            patience=50
        ) 


# import pandas as pd 
# import read_tess_data
# from read_tess_data import read_light_curve, light_curve_to_power_spectrum
# import h5py
# import numpy as np
# from run_mlp import apply_min_max_scaling
# import torch 
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn as nn
# import wandb
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, r2_score
# import run_cnn
# from data import read_hdf5_data

# torch.manual_seed(42)
# np.random.seed(42)

# def read_data():
#     df_iacob1 = pd.read_csv('/mnt/sdceph/users/rzhang/iacob1.csv')
#     tic_id = df_iacob1['TIC_ID']
#     teff = df_iacob1['Teff']
#     logg = df_iacob1['logg']
#     Msp = df_iacob1['Msp']
#     return tic_id, teff, logg, Msp

# def save_power_freq_info(h5_file_path):
#     tic_id, teff, logg, Msp = read_data()
#     db = read_tess_data.Database('/mnt/home/neisner/ceph/latte/output_LATTE/tess_database.db')
#     with h5py.File(h5_file_path, 'w') as h5f:
#         for tic_id, t, g, m in zip(tic_id, teff, logg, Msp):
#             sectorids, lcpaths, tppaths = db.search(tic_id)
#             if lcpaths != 0:
#                 obs_id = 0
#                 for filepath in lcpaths:
#                     print('looking for filepath: ', filepath)
#                     if read_light_curve(filepath) is not None:
#                         time, flux = read_light_curve(filepath)
#                         freq, power = light_curve_to_power_spectrum(time, flux)
#                         dataset_name = f'TIC_{tic_id}_{obs_id}_Power'
#                         h5f.create_dataset(dataset_name, data=power)
#                         if 'Frequency' not in h5f:
#                             h5f.create_dataset('Frequency', data=freq)
#                         h5f[dataset_name].attrs['TIC_ID'] = tic_id
#                         h5f[dataset_name].attrs['Teff'] = t
#                         h5f[dataset_name].attrs['logg'] = g
#                         h5f[dataset_name].attrs['Msp'] = m
#                         obs_id += 1

# def calculate_lum_from_teff_logg(Teff, logg, tensor):
#     if tensor == False:
#         log_L_sun=np.log10((5777)**4.0/(274*100))  # Used for Spectroscopic HRD
#         Teff_K = Teff * 1000
#         logL_solar = 4*np.log10(Teff_K)-logg-log_L_sun
#         # G = 6.67e-8
#         # sigma = 5.67e-5
#         # g_in_solar_mass = 1.989e33
#         # erg_in_solar_lum = 3.826e33
#         # g = 10 ** logg
#         # # if np.isinf(g).any():
#         # #     g[np.isinf(g)] = 1e6
#         # Msp_cgs = Msp * g_in_solar_mass
#         # Teff_K = Teff * 1000
#         # L = 4 * np.pi * G * Msp_cgs * sigma * Teff_K**4 / g
#         # L_solar = L / erg_in_solar_lum
#         # logL_solar = np.log10(L_solar)
#     else:
#         log_L_sun=torch.tensor((np.log10((5777)**4.0/(274*100))), dtype=torch.float64)
#         Teff_K = Teff * 1000
#         logL_solar = 4*torch.log10(Teff_K)-logg-log_L_sun
#         # G = torch.tensor(6.67e-8, dtype=torch.float64)
#         # sigma = torch.tensor(5.67e-5, dtype=torch.float64)
#         # g_in_solar_mass = torch.tensor(1.989e33, dtype=torch.float64)
#         # erg_in_solar_lum = torch.tensor(3.826e33, dtype=torch.float64)
#         # g = 10 ** logg
#         # # print('g', g)
#         # # if torch.isinf(g).any():
#         # #     g[torch.isinf(g)] = torch.tensor(1e6, dtype=torch.float64)
#         # Msp_cgs = Msp * g_in_solar_mass
#         # Teff_K = Teff * 1000
#         # L = 4 * torch.pi * G * Msp_cgs * sigma * Teff_K**4 / g
#         # L_solar = L / erg_in_solar_lum
#         # logL_solar = torch.log10(L_solar)
#     return logL_solar

# def normalize_data(data):
#     mean = np.mean(data)
#     std = np.std(data)
#     return (data - mean) / std, mean, std

# def preprocess_data(power, Teff, logg, Msp, freq):
#     Teff = [float(t) for t in Teff]
#     logg = [float(l) for l in logg]
#     Msp = [float(m) for m in Msp]
    
#     Teff_K = [t * 1000 for t in Teff]
#     logTeff_K = np.log10(Teff_K)

#     Teff_norm, Teff_mean, Teff_std = normalize_data(np.array(Teff))
#     logTeff_norm, logTeff_mean, logTeff_std = normalize_data(np.array(logTeff_K))
#     logg_norm, logg_mean, logg_std = normalize_data(np.array(logg))
#     Msp_norm, Msp_mean, Msp_std = normalize_data(np.array(Msp))
#     logL = calculate_lum_from_teff_logg(np.array(Teff), np.array(logg), False)
#     logL_norm, logL_mean, logL_std = normalize_data(logL)

#     # log-transform and scale the power spectrum
#     logpower = power.apply(lambda x: np.log10(x).tolist())
#     scaled_power = apply_min_max_scaling(logpower)
#     power_tensor = torch.tensor(np.array(scaled_power.tolist(), dtype=np.float32))
    
#     # normalized labels tensor for Teff, logg, and Msp
#     # labels_tensor = torch.tensor(list(zip(Teff_norm, logg_norm, Msp_norm)), dtype=torch.float32)
#     # labels_tensor = torch.tensor(list(zip(Teff_norm, logg_norm)), dtype=torch.float32)
#     labels_tensor = torch.tensor(list(zip(Teff_norm, logL_norm)), dtype=torch.float32)
#     # labels_tensor = torch.tensor(list(zip(logTeff_norm, logL_norm)), dtype=torch.float32)

#     # # TRY ONLY PREDICTING TEFF
#     # labels_tensor = torch.tensor(list(zip(Teff_norm)), dtype=torch.float32)

#     return power_tensor, labels_tensor, (Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std, logTeff_mean, logTeff_std)

# def create_dataloaders(power_tensor, labels_tensor, batch_size):
#     dataset = TensorDataset(power_tensor, labels_tensor)
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader, train_dataset, test_dataset

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
#             nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
#             # nn.BatchNorm1d(num_channels * 2), 
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
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
#             # nn.Linear(128, 64),  
#             # nn.ReLU(),
#             nn.Linear(128, output_size),
#         )
    
#     def forward(self, x):
#         x = self.conv_layers(x) # process inputs through convolutional layers
#         x = self.flatten(x) # flattens output of convolutional layers
#         logits = self.fc_layers(x) # passes flattened output through fully connected layers to produce logits for each class
#         return logits # raw, unnormalized scores for each class

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )

#     def forward(self, x):
#         output = self.linear_relu_stack(x)
#         return output

# def weighted_mse_loss(pred_tensor, actual_tensor, weights):  
#     se = (pred_tensor - actual_tensor) ** 2
#     weighted_se = se * weights
#     weighted_se_mean = weighted_se.mean()
#     return weighted_se_mean

# def scaled_mse_loss(pred_tensor, actual_tensor, scales):  
#     se = (pred_tensor - actual_tensor) ** 2
#     scaled_se = se / (scales ** 2)
#     scaled_se_mean = scaled_se.mean()
#     return scaled_se_mean

# def train_loop(dataloader, model, loss_fn, optimizer, epoch, norm_params):
#     model.train()
#     total_loss = 0
#     # Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
#     # weights = torch.tensor([10.0, 1.0], dtype=torch.float32, device='cuda')
#     # scales = torch.tensor([Teff_std, logL_std], dtype=torch.float32, device='cuda')
#     for X, y in dataloader:
#         X, y = X.cuda().unsqueeze(1), y.cuda() #UNSQUEEZE IF CNN
#         pred = model(X)        
#         # # apply exponential transformation
#         # pred_exp = torch.exp(pred).cuda()
#         # Teff_pred, logg_pred, Msp_pred = pred_exp[:, 0] * Teff_std + Teff_mean, pred_exp[:, 1] * logg_std + logg_mean, pred_exp[:, 2] * Msp_std + Msp_mean
#         # Teff_actual, logg_actual, Msp_actual = y[:, 0] * Teff_std + Teff_mean, y[:, 1] * logg_std + logg_mean, y[:, 2] * Msp_std + Msp_mean
        
#         # Teff_pred = Teff_pred.to(dtype=torch.float64, device='cuda')
#         # logg_pred = logg_pred.to(dtype=torch.float64, device='cuda')
#         # Msp_pred = Msp_pred.to(dtype=torch.float64, device='cuda')
#         # Teff_actual = Teff_actual.to(dtype=torch.float64, device='cuda')
#         # logg_actual = logg_actual.to(dtype=torch.float64, device='cuda')
#         # Msp_actual = Msp_actual.to(dtype=torch.float64, device='cuda')

#         # pred_logL = calculate_lum_from_teff_logg(Teff_pred, logg_pred, Msp_pred, True).unsqueeze(-1)
#         # actual_logL = calculate_lum_from_teff_logg(Teff_actual, logg_actual, Msp_actual, True).unsqueeze(-1)

#         # pred_tensor = torch.cat([pred_exp[:, 0].unsqueeze(-1), (pred_logL - logL_mean) / logL_std], dim=1)
#         # actual_tensor = torch.cat([y[:, 0].unsqueeze(-1), (actual_logL - logL_mean) / logL_std], dim=1)
#         # # pred_tensor = torch.cat([Teff_pred.unsqueeze(-1), pred_logL], dim=1)
#         # # actual_tensor = torch.cat([Teff_actual.unsqueeze(-1), actual_logL], dim=1)
#         # # loss = weighted_mse_loss(pred_tensor, actual_tensor, weights)
#         # # loss = scaled_mse_loss(pred_tensor, actual_tensor, scales)
#         # # loss = loss_fn(pred_tensor, actual_tensor)
#         # # loss = loss_fn(pred_logL, actual_logL)
#         loss = loss_fn(pred, y)
#         # loss = weighted_mse_loss(pred, y, weights)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     #     print('epoch in train loop: ', epoch)
#     #     print('predicted', pred)
#     #     print('actual', y)
#     #     print('loss', loss.item())
#     # print('dataloader length', len(dataloader))
#     avg_loss = total_loss / len(dataloader)
#     wandb.log({"train_loss": avg_loss, "epoch": epoch})
#     print(f'Training loss: {avg_loss}')

# def denormalize_data(norm_data, means, stds):
#     denorm_data = []
#     for data in norm_data:
#         denorm_sublist = [(value * std + mean) for value, mean, std in zip(data, means, stds)]
#         denorm_data.append(denorm_sublist)
#     return np.array(denorm_data)

# def test_loop(dataloader, model, loss_fn, epoch, norm_params):
#     model.eval()
#     total_loss = 0
#     Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std, logTeff_mean, logTeff_std = norm_params
#     all_preds = []
#     all_ys = []
#     # weights = torch.tensor([10.0, 1.0], dtype=torch.float32, device='cuda')
#     # scales = torch.tensor([Teff_std, logL_std], dtype=torch.float32, device='cuda')
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.cuda().unsqueeze(1), y.cuda()
#             pred = model(X)
#             # # exponential transformation
#             # pred_exp = torch.exp(pred).cuda()
#             # Teff_pred, logg_pred, Msp_pred = pred_exp[:, 0] * Teff_std + Teff_mean, pred_exp[:, 1] * logg_std + logg_mean, pred_exp[:, 2] * Msp_std + Msp_mean
#             # Teff_actual, logg_actual, Msp_actual = y[:, 0] * Teff_std + Teff_mean, y[:, 1] * logg_std + logg_mean, y[:, 2] * Msp_std + Msp_mean
#             # Teff_pred = Teff_pred.to(dtype=torch.float64, device='cuda')
#             # logg_pred = logg_pred.to(dtype=torch.float64, device='cuda')
#             # Msp_pred = Msp_pred.to(dtype=torch.float64, device='cuda')
#             # Teff_actual = Teff_actual.to(dtype=torch.float64, device='cuda')
#             # logg_actual = logg_actual.to(dtype=torch.float64, device='cuda')
#             # Msp_actual = Msp_actual.to(dtype=torch.float64, device='cuda')
#             # pred_logL = calculate_lum_from_teff_logg(Teff_pred, logg_pred, Msp_pred, True).unsqueeze(-1)
#             # actual_logL = calculate_lum_from_teff_logg(Teff_actual, logg_actual, Msp_actual, True).unsqueeze(-1)
#             # pred_tensor = torch.cat([pred_exp[:, 0].unsqueeze(-1), (pred_logL - logL_mean) / logL_std], dim=1)
#             # actual_tensor = torch.cat([y[:, 0].unsqueeze(-1), (actual_logL - logL_mean) / logL_std], dim=1)
#             # # pred_tensor = torch.cat([Teff_pred.unsqueeze(-1), pred_logL], dim=1)
#             # # actual_tensor = torch.cat([Teff_actual.unsqueeze(-1), actual_logL], dim=1)
#             # # print('pred_tensor', pred_tensor)
#             # # total_loss += weighted_mse_loss(pred_tensor, actual_tensor, weights)
#             # # total_loss += scaled_mse_loss(pred_tensor, actual_tensor, scales)
#             # # total_loss += loss_fn(pred_tensor, actual_tensor)
#             # # total_loss += loss_fn(pred_logL, actual_logL)
#             total_loss += loss_fn(pred, y).item()
#             # total_loss += weighted_mse_loss(pred, y, weights).item()
#             all_preds.extend(pred.cpu().numpy())
#             all_ys.extend(y.cpu().numpy())
#     avg_loss = total_loss / len(dataloader)
#     wandb.log({"test_loss": avg_loss, "epoch": epoch})
#     print(f'Average loss: {avg_loss}')

#     # # PREDICT ONLY TEFF
#     # all_preds = denormalize_data(all_preds, [Teff_mean], [Teff_std])
#     # all_ys = denormalize_data(all_ys, [Teff_mean], [Teff_std])

#     # all_preds = denormalize_data(all_preds, [Teff_mean, logg_mean], [Teff_std, logg_std])
#     # all_ys = denormalize_data(all_ys, [Teff_mean, logg_mean], [Teff_std, logg_std])
    
#     # all_preds = denormalize_data(all_preds, [Teff_mean, logg_mean, Msp_mean], [Teff_std, logg_std, Msp_std])
#     # all_ys = denormalize_data(all_ys, [Teff_mean, logg_mean, Msp_mean], [Teff_std, logg_std, Msp_std])

#     all_preds = denormalize_data(all_preds, [Teff_mean, logL_mean], [Teff_std, logL_std])
#     all_ys = denormalize_data(all_ys, [Teff_mean, logL_mean], [Teff_std, logL_std])

#     # all_preds = denormalize_data(all_preds, [logTeff_mean, logL_mean], [logTeff_std, logL_std])
#     # all_ys = denormalize_data(all_ys, [logTeff_mean, logL_mean], [logTeff_std, logL_std])
    
#     # print('ALL TEFF PRINTS ARE ACTUALLY LOGTEFF PRINTS')
#     print('epoch in test loop: ', epoch)
#     # print('all ys', all_ys)
#     # print('all preds', all_preds)
#     print('actual Teff', all_ys[:, 0])
#     print('pred Teff', all_preds[:, 0])
#     # actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), False)
#     # print('actual logL', actual_logL)
#     # pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), False)
#     # print('pred logL', pred_logL)

#     print('actual logL', all_ys[:, 1])
#     print('pred logL', all_preds[:, 1])

#     plt.figure(figsize=(10, 6))
#     plt.scatter(all_ys[:, 0], all_preds[:, 0], alpha=0.3)
#     plt.xlabel('Actual Teff')
#     plt.ylabel('Predicted Teff')
#     plt.title('Predicted vs Actual Teff')
#     plt.plot([all_ys[:, 0].min(), all_ys[:, 0].max()], [all_ys[:, 0].min(), all_ys[:, 0].max()], 'r')
#     plt.grid(True)
#     plt.savefig("pred_vs_act_Teff.png")
#     wandb.log({"CNN Predicted vs Actual Teff": wandb.Image("pred_vs_act_Teff.png", caption="Predictions vs. Actual Teff at Epoch {}".format(epoch))})
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
#     plt.xlabel('Actual logL')
#     plt.ylabel('Predicted logL')
#     plt.title('Predicted vs Actual logL')
#     plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
#     plt.grid(True)
#     plt.savefig("pred_vs_act_logL.png")
#     wandb.log({"CNN Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
#     plt.close()

#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(all_ys[:, 1], all_preds[:, 1], alpha=0.3)
#     # plt.xlabel('Actual logg')
#     # plt.ylabel('Predicted logg')
#     # plt.title('Predicted vs Actual logg')
#     # plt.plot([all_ys[:, 1].min(), all_ys[:, 1].max()], [all_ys[:, 1].min(), all_ys[:, 1].max()], 'r')
#     # plt.grid(True)
#     # plt.savefig("pred_vs_act_logg.png")
#     # wandb.log({"CNN Predicted vs Actual logg": wandb.Image("pred_vs_act_logg.png", caption="Predictions vs. Actual logg at Epoch {}".format(epoch))})
#     # plt.close()

#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(all_ys[:, 2], all_preds[:, 2], alpha=0.3)
#     # plt.xlabel('Actual Msp')
#     # plt.ylabel('Predicted Msp')
#     # plt.title('Predicted vs Actual Msp')
#     # plt.plot([all_ys[:, 2].min(), all_ys[:, 2].max()], [all_ys[:, 2].min(), all_ys[:, 2].max()], 'r')
#     # plt.grid(True)
#     # plt.savefig("pred_vs_act_Msp.png")
#     # wandb.log({"CNN Predicted vs Actual Msp": wandb.Image("pred_vs_act_Msp.png", caption="Predictions vs. Actual Msp at Epoch {}".format(epoch))})
#     # plt.close()

#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(actual_logL, pred_logL, alpha=0.3)
#     # plt.xlabel('Actual logL')
#     # plt.ylabel('Predicted logL')
#     # plt.title('Predicted vs Actual Spectroscopic logL')
#     # plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
#     # plt.grid(True)
#     # plt.savefig("pred_vs_act_logL.png")
#     # wandb.log({"CNN Predicted vs Actual logL": wandb.Image("pred_vs_act_logL.png", caption="Predictions vs. Actual logL at Epoch {}".format(epoch))})
#     # plt.close()

#     # if ~np.isnan(pred_logL).any():
#     print('epoch: ', epoch)
#     mse_logL = mean_squared_error(all_ys[:, 1], all_preds[:, 1])
#     # mse_logL = mean_squared_error(actual_logL, pred_logL)
#     print('MSE of logL', mse_logL)
#     mse_Teff = mean_squared_error(all_ys[:, 0], all_preds[:, 0])
#     print('MSE of Teff', mse_Teff)
#     # mse_logg = mean_squared_error(all_ys[:, 1], all_preds[:, 1])
#     # print('MSE of logg', mse_logg)
    
#     r2_logL = r2_score(all_ys[:, 1], all_preds[:, 1])
#     # r2_logL = r2_score(actual_logL, pred_logL)
#     print('R2 score of logL', r2_logL)
#     r2_Teff = r2_score(all_ys[:, 0], all_preds[:, 0])
#     print('R2 score of Teff', r2_Teff)
#     # r2_logg = r2_score(all_ys[:, 1], all_preds[:, 1])
#     # print('R2 score of logg', r2_logg)

#     # residuals = all_preds[:, 0] - all_ys[:, 0]
#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(all_ys[:, 0], residuals, alpha=0.3)
#     # plt.xlabel('Actual Teff')
#     # plt.ylabel('Residuals')
#     # plt.title('Residuals Plot Teff')
#     # plt.hlines(y=0, xmin=all_ys[:, 0].min(), xmax=all_ys[:, 0].max(), colors='r')
#     # plt.grid(True)
#     # plt.savefig("residuals_Teff.png")
#     # wandb.log({"Residuals Teff": wandb.Image("residuals_Teff.png", caption="Residuals Teff at Epoch {}".format(epoch))})
#     # plt.close()

#     # residuals = all_preds[:, 1] - all_ys[:, 1]
#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(all_ys[:, 1], residuals, alpha=0.3)
#     # plt.xlabel('Actual logL')
#     # plt.ylabel('Residuals')
#     # plt.title('Residuals Plot logL')
#     # plt.hlines(y=0, xmin=all_ys[:, 1].min(), xmax=all_ys[:, 1].max(), colors='r')
#     # plt.grid(True)
#     # plt.savefig("residuals_logL.png")
#     # wandb.log({"Residuals logL": wandb.Image("residuals_logL.png", caption="Residuals logL at Epoch {}".format(epoch))})
#     # plt.close()
#     return avg_loss

# # def get_spectroscopic_lum_info(all_ys, all_preds):
# #     print('all ys', all_ys)
# #     print('all preds', all_preds)
# #     actual_logL = calculate_lum_from_teff_logg(np.array(all_ys[:, 0], dtype=np.float64), np.array(all_ys[:, 1], dtype=np.float64), np.array(all_ys[:, 2], dtype=np.float64))
# #     print('actual logL', actual_logL)
# #     pred_logL = calculate_lum_from_teff_logg(np.array(all_preds[:, 0], dtype=np.float64), np.array(all_preds[:, 1], dtype=np.float64), np.array(all_preds[:, 2], dtype=np.float64))
# #     print('pred logL', pred_logL)
# #     plt.figure(figsize=(10, 6))
# #     plt.scatter(actual_logL, pred_logL, alpha=0.3)
# #     plt.xlabel('Actual logL')
# #     plt.ylabel('Predicted logL')
# #     plt.title('Predicted vs Actual Spectroscopic logL')
# #     plt.plot([min(actual_logL), max(actual_logL)], [min(actual_logL), max(actual_logL)], 'r')
# #     plt.grid(True)
# #     plt.savefig("pred_vs_act_logL.png")
# #     plt.close()

# #     mse = mean_squared_error(actual_logL, pred_logL)
# #     print('MSE', mse)
    
# #     r2 = r2_score(actual_logL, pred_logL)
# #     print('R2 score', r2)

# if __name__ == '__main__':
#     wandb.init(project="lightcurve-to-spectra-ml-regression", entity="rczhang")
#     h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression.h5'
#     # save_power_freq_info(h5_file_path)
#     power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')  
#     print('len teff', len(Teff))
#     power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, frequencies)
#     print('normalization params', normalization_params)
#     batch_size = 32
#     epochs = 10000
#     train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size)
#     loss_fn = nn.MSELoss().cuda()
#     # model = MLP(input_size=len(power.iloc[0]), output_size=3).cuda()
#     num_channels = 32
#     input_size = len(power.iloc[0])
#     model = CNN1D(num_channels, 2, input_size).cuda()
#     # model.load_state_dict(torch.load('best_reg_colorful_serenity_82.pth', map_location='cuda'))
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
#     best_loss = float('inf')
#     patience = 300 
#     patience_counter = 0
#     for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(train_loader, model, loss_fn, optimizer, t, normalization_params)
#         current_loss = test_loop(test_loader, model, loss_fn, t, normalization_params)
#         scheduler.step(current_loss)
#         if current_loss < best_loss:
#             best_loss = current_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), f"best_regression_{wandb.run.name}.pth")
#         else:
#             patience_counter += 1
#         if patience_counter >= patience:
#             print('Early stopping triggered')
#             break 
#     # get_spectroscopic_lum_info(all_ys, all_preds)
#     print("Done!")
#     wandb.finish()