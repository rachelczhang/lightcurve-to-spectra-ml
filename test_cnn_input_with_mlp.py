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
        """
        input_size: size of input (length of power spectrum)
        hidden_sizes: list of hidden layer sizes, e.g., [512, 256] for 2 hidden layers
        output_size: number of outputs (2 for Teff+logg or Teff+logL)
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input in case it has extra dimensions
        x = x.view(x.size(0), -1)
        return self.network(x)

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

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.cuda(), y.cuda()  # No need to unsqueeze for MLP
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
    
    return r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl

def calculate_summary_statistics(results_for_arch):
    """Calculate summary statistics for an architecture across multiple seeds"""
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results_for_arch)
    
    # Calculate statistics for each metric
    stats = {}
    metrics = ['R2_Teff', 'R2_logg', 'R2_logL', 'RMSE_Teff', 'RMSE_logg', 'RMSE_logL']
    
    for metric in metrics:
        values = df[metric].values
        stats[f'{metric}_mean'] = np.mean(values)
        stats[f'{metric}_median'] = np.median(values)
        stats[f'{metric}_std'] = np.std(values)
        stats[f'{metric}_min'] = np.min(values)
        stats[f'{metric}_max'] = np.max(values)
        stats[f'{metric}_range'] = np.max(values) - np.min(values)
        stats[f'{metric}_q10'] = np.percentile(values, 10)
        stats[f'{metric}_q90'] = np.percentile(values, 90)
        stats[f'{metric}_80pct_interval'] = stats[f'{metric}_q90'] - stats[f'{metric}_q10']
    
    # Add other useful statistics
    stats['avg_epochs'] = np.mean(df['Epochs'].values)
    stats['avg_train_time_min'] = np.mean(df['Train_Time_min'].values)
    
    return stats

def run_mlp_comparison_study(predict_target='teff_logg', num_seeds=10):
    """
    Test MLP architectures on CNN input data to demonstrate CNNs are better for sequential data
    
    predict_target: 'teff_logg' or 'teff_logl'
    num_seeds: number of random seeds to test per architecture
    """
    # Load data (same as CNN study)
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    input_size = len(power.iloc[0])
    print(f"Input size (power spectrum length): {input_size}")
    
    # Use same seeds as CNN study for fair comparison
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627][:num_seeds]
    
    # Define MLP architectures to test
    # Format: (hidden_sizes, description)
    architectures = [
        # Single hidden layer architectures 
        ([1024], "MLP-1024"),
        ([2048], "MLP-2048"),
        ([4096], "MLP-4096"),
        ([8192], "MLP-8192"),
        
        # Two hidden layer architectures - expand then contract
        ([8192, 2048], "MLP-8192-2048"),
        ([8192, 1024], "MLP-8192-1024"),
        ([4096, 1024], "MLP-4096-1024"),
        ([2048, 512], "MLP-2048-512"),
        
        # Three hidden layer architectures - gradual tapering
        ([8192, 2048, 512], "MLP-8192-2048-512"),
        ([4096, 2048, 512], "MLP-4096-2048-512"),
        ([8192, 4096, 1024], "MLP-8192-4096-1024"),
    ]
    
    all_results = []
    summary_results = []
    batch_size = 32
    max_epochs = 500
    patience = 50
    
    target_name = "Teff+logg" if predict_target == 'teff_logg' else "Teff+logL"
    print(f"Running MLP comparison study for {target_name} prediction...")
    print(f"Testing {num_seeds} random seeds per architecture...")
    print("Architecture | Params | Avg Epochs | Avg Time(min) | R²(Teff) Med[80%CI] | R²(logg) Med[80%CI] | R²(logL) Med[80%CI]")
    print("-" * 120)
    
    for arch_idx, (hidden_sizes, arch_desc) in enumerate(architectures):
        arch_start_time = time.time()
        results_for_arch = []
        
        # Calculate number of parameters
        model_temp = MLP(input_size, hidden_sizes, 2).cuda()
        num_params = count_parameters(model_temp)
        del model_temp  # Free memory
        
        print(f"\nRunning architecture {arch_idx+1}/{len(architectures)}: {arch_desc}")
        
        # Run multiple seeds for this architecture
        for seed_idx, seed in enumerate(seeds):
            # Set all seeds for this run
            set_all_seeds(seed)
            
            # Create data loaders with this seed
            train_loader, test_loader, _ = create_dataloaders(power_tensor, labels_tensor, batch_size, seed)
            
            # Create model
            model = MLP(input_size, hidden_sizes, 2).cuda()
            
            # Training setup
            loss_fn = nn.MSELoss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
            
            # Train model
            start_time = time.time()
            epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, normalization_params)
            train_time = time.time() - start_time
            
            # Evaluate model
            r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl = evaluate_model(model, test_loader, normalization_params, predict_target)
            
            # Store results
            result = {
                'Architecture': arch_desc,
                'Seed': seed,
                'Seed_Index': seed_idx + 1,
                'Hidden_Sizes': str(hidden_sizes),
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
            results_for_arch.append(result)
            all_results.append(result)
            
            # Print progress for this seed
            if (seed_idx + 1) % 5 == 0 or seed_idx == 0:
                print(f"  Seed {seed_idx+1:2}/{num_seeds}: R²(Teff)={r2_teff:.3f}, R²(logL)={r2_logl:.3f}")
        
        # Calculate summary statistics for this architecture
        stats = calculate_summary_statistics(results_for_arch)
        
        # Add architecture info to summary
        summary_result = {
            'Architecture': arch_desc,
            'avg_epochs': stats['avg_epochs'],
            'avg_train_time_min': stats['avg_train_time_min'],
            'Parameters': num_params,
            'Hidden_Sizes': str(hidden_sizes),
            'Predict_Target': predict_target,
            'Num_Seeds': num_seeds,
            **{k: v for k, v in stats.items() if k not in ['avg_epochs', 'avg_train_time_min']}
        }
        summary_results.append(summary_result)
        
        # Print summary for this architecture
        arch_total_time = time.time() - arch_start_time
        print(f"{arch_desc:15} | {num_params:6} | {stats['avg_epochs']:8.1f} | {arch_total_time/60:8.1f} | "
              f"{stats['R2_Teff_median']:.3f}[{stats['R2_Teff_q10']:.3f}-{stats['R2_Teff_q90']:.3f}] | "
              f"{stats['R2_logg_median']:.3f}[{stats['R2_logg_q10']:.3f}-{stats['R2_logg_q90']:.3f}] | "
              f"{stats['R2_logL_median']:.3f}[{stats['R2_logL_q10']:.3f}-{stats['R2_logL_q90']:.3f}]")
    
    # Save detailed results to CSV
    detailed_df = pd.DataFrame(all_results)
    detailed_filename = f'mlp_vs_cnn_input_{predict_target}_detailed_results.csv'
    detailed_df.to_csv(detailed_filename, index=False)
    
    # Save summary results to CSV
    summary_df = pd.DataFrame(summary_results)
    summary_filename = f'mlp_vs_cnn_input_{predict_target}_summary_results.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    print(f"\nDetailed results saved to {detailed_filename}")
    print(f"Summary results saved to {summary_filename}")
    
    # Analysis of results
    print(f"\n" + "="*80)
    print(f"MLP PERFORMANCE ANALYSIS FOR {target_name.upper()} PREDICTION")
    print("="*80)
    
    # Calculate combined score: average of median R² for Teff and logL
    summary_df['Combined_R2_Score'] = (summary_df['R2_Teff_median'] + summary_df['R2_logL_median']) / 2
    
    # Find best MLP architecture
    best_mlp = summary_df.loc[summary_df['Combined_R2_Score'].idxmax()]
    
    print(f"\nBest MLP architecture (highest combined median R² Teff+logL): {best_mlp['Architecture']}")
    print(f"  Combined Score (Teff+logL): {best_mlp['Combined_R2_Score']:.4f}")
    print(f"  Median R² (Teff): {best_mlp['R2_Teff_median']:.4f} [80% CI: {best_mlp['R2_Teff_q10']:.4f}-{best_mlp['R2_Teff_q90']:.4f}]")
    print(f"  Median R² (logL): {best_mlp['R2_logL_median']:.4f} [80% CI: {best_mlp['R2_logL_q10']:.4f}-{best_mlp['R2_logL_q90']:.4f}]")
    print(f"  Parameters: {best_mlp['Parameters']:,}")
    print(f"  Hidden layers: {best_mlp['Hidden_Sizes']}")
    
    if predict_target == 'teff_logg':
        print(f"  Median R² (logg): {best_mlp['R2_logg_median']:.4f} [80% CI: {best_mlp['R2_logg_q10']:.4f}-{best_mlp['R2_logg_q90']:.4f}]")
    
    # Show top 5 architectures for comparison
    summary_df_sorted = summary_df.sort_values('Combined_R2_Score', ascending=False)
    print(f"\nTop 5 MLP architectures by combined median R² (Teff+logL):")
    for i in range(min(5, len(summary_df_sorted))):
        arch = summary_df_sorted.iloc[i]
        print(f"  {i+1}. {arch['Architecture']}: Combined = {arch['Combined_R2_Score']:.4f} "
              f"[Teff: {arch['R2_Teff_median']:.4f}, logL: {arch['R2_logL_median']:.4f}]")
    
    # Performance summary statistics
    print(f"\nOverall MLP Performance Summary:")
    print(f"  Best Combined R² Score: {summary_df['Combined_R2_Score'].max():.4f}")
    print(f"  Mean Combined R² Score: {summary_df['Combined_R2_Score'].mean():.4f} ± {summary_df['Combined_R2_Score'].std():.4f}")
    print(f"  Range of Combined R² Scores: {summary_df['Combined_R2_Score'].min():.4f} - {summary_df['Combined_R2_Score'].max():.4f}")
    
    print(f"\n  Best Individual Metrics:")
    print(f"    R² (Teff): {summary_df['R2_Teff_median'].max():.4f}")
    print(f"    R² (logL): {summary_df['R2_logL_median'].max():.4f}")
    if predict_target == 'teff_logg':
        print(f"    R² (logg): {summary_df['R2_logg_median'].max():.4f}")
    
    print(f"\n  NOTE: Compare these results with CNN ablation study results to see")
    print(f"        that CNNs perform better on this sequential power spectrum data.")
    
    return all_results, summary_results

if __name__ == '__main__':
    # Run MLP comparison for both prediction targets
    print("=" * 50)
    print("MLP COMPARISON STUDY 1: Predicting Teff + logg")
    print("=" * 50)
    detailed_results_1, summary_results_1 = run_mlp_comparison_study('teff_logg', num_seeds=10)
    
    print("\n" + "=" * 50)
    print("MLP COMPARISON STUDY 2: Predicting Teff + logL")
    print("=" * 50)
    detailed_results_2, summary_results_2 = run_mlp_comparison_study('teff_logl', num_seeds=10) 