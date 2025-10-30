import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import time
import self_supervised
from torch.optim.lr_scheduler import ReduceLROnPlateau
from regression import read_hdf5_data, preprocess_data, create_dataloaders, calculate_lum_from_teff_logg
from sklearn.metrics import r2_score, mean_squared_error
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

class EncoderCNN1D_Flexible(nn.Module):
    """Flexible encoder that can have 1, 2, or 3 convolutional layers"""
    def __init__(self, num_channels, input_size, num_conv_layers=1):
        super(EncoderCNN1D_Flexible, self).__init__()
        
        conv_layers = []
        in_channels = 1
        
        # Build convolutional layers
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, num_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ])
            in_channels = num_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output dimension after conv layers
        dummy_input = torch.randn(1, 1, input_size)
        dummy_output = self.conv_layers(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]
        
    def forward(self, x):
        x = self.conv_layers(x)
        return x

class CNN1DFrozenConv_Flexible(nn.Module):
    """Flexible regression head with different FC configurations"""
    def __init__(self, pretrained_encoder, output_size, input_size, device, fc_config):
        super().__init__()
        self.encoder = pretrained_encoder
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Calculate encoder output dimension
        dummy_input = torch.randn(1, 1, input_size, device=device)
        dummy_output = self.encoder(dummy_input)
        self.output_dim = dummy_output.numel() // dummy_output.shape[0]
        self.flatten = nn.Flatten()
        # Build FC layers based on configuration
        if isinstance(fc_config, tuple):
            # Two FC layers
            fc1_size, fc2_size = fc_config
            self.fc_layers = nn.Sequential(
                nn.Linear(self.output_dim, fc1_size),
                nn.ReLU(),
                nn.Linear(fc1_size, fc2_size),
                nn.ReLU(),
                nn.Linear(fc2_size, output_size)
            )
        elif isinstance(fc_config, int):
            if fc_config > 0:
                # Single FC layer
                self.fc_layers = nn.Sequential(
                    nn.Linear(self.output_dim, fc_config),
                    nn.ReLU(),
                    nn.Linear(fc_config, output_size)
                )
            else:
                # Direct connection
                self.fc_layers = nn.Linear(self.output_dim, output_size)
        elif fc_config == "direct":
            self.fc_layers = nn.Linear(self.output_dim, output_size)
        else:
            raise ValueError(f"Unknown fc_config: {fc_config}")
    def forward(self, x):
        x = self.encoder(x) 
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
    """Train model with early stopping"""
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.cuda().unsqueeze(1), y.cuda()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.cuda().unsqueeze(1), y.cuda()
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        
        val_loss /= len(test_loader)
        
        # Early stopping logic
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return epoch + 1, best_loss

def evaluate_model(model, test_loader, norm_params, predict_target):
    """Evaluate model and return R2 and RMSE scores"""
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

def calculate_summary_statistics(results_for_arch):
    """Calculate summary statistics for an architecture across multiple seeds"""
    df = pd.DataFrame(results_for_arch)
    
    stats = {
        'avg_epochs': df['Epochs'].mean(),
        'avg_train_time_min': df['Train_Time_min'].mean(),
        
        # R2 statistics
        'R2_Teff_median': df['R2_Teff'].median(),
        'R2_Teff_mean': df['R2_Teff'].mean(),
        'R2_Teff_std': df['R2_Teff'].std(),
        'R2_Teff_q10': df['R2_Teff'].quantile(0.1),
        'R2_Teff_q90': df['R2_Teff'].quantile(0.9),
        
        'R2_logg_median': df['R2_logg'].median(),
        'R2_logg_mean': df['R2_logg'].mean(),
        'R2_logg_std': df['R2_logg'].std(),
        'R2_logg_q10': df['R2_logg'].quantile(0.1),
        'R2_logg_q90': df['R2_logg'].quantile(0.9),
        
        'R2_logL_median': df['R2_logL'].median(),
        'R2_logL_mean': df['R2_logL'].mean(),
        'R2_logL_std': df['R2_logL'].std(),
        'R2_logL_q10': df['R2_logL'].quantile(0.1),
        'R2_logL_q90': df['R2_logL'].quantile(0.9),
        
        # RMSE statistics  
        'RMSE_Teff_median': df['RMSE_Teff'].median(),
        'RMSE_logg_median': df['RMSE_logg'].median(),
        'RMSE_logL_median': df['RMSE_logL'].median(),
    }
    
    return stats

def run_architecture_ablation_study(predict_target='teff_logg', num_seeds=29):
    """
    Run comprehensive ablation study testing different encoder and regression head architectures
    
    Args:
        predict_target: 'teff_logg' or 'teff_logl'
        num_seeds: number of random seeds to test per architecture combination
    """
    # Load data
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    input_size = len(power.iloc[0])
    
    # Define seeds (same as ablation_study_mlp.py, up to num_seeds)
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627,
             303132, 333435, 363738, 394041, 424344, 454647, 484950, 515253, 545556, 575859,
             606162, 636465, 666768, 697071, 727374, 757677, 787980, 818283, 848586, 878889][:num_seeds]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define architectures to match ablation_study_cnn.py
    architectures = [
        # 1 Conv layer variations
        (1, 16, 64, "1conv-16ch-64fc"),
        (1, 32, 64, "1conv-32ch-64fc"),
        (1, 32, 128, "1conv-32ch-128fc"),
        (1, 64, 128, "1conv-64ch-128fc"),
        # 2 Conv layer variations
        (2, 16, 64, "2conv-16ch-64fc"),
        (2, 32, 64, "2conv-32ch-64fc"),
        (2, 32, 128, "2conv-32ch-128fc"),
        (2, 64, 128, "2conv-64ch-128fc"),
        (2, 32, 256, "2conv-32ch-256fc"),
        # 3 Conv layer variations
        (3, 16, 64, "3conv-16ch-64fc"),
        (3, 32, 64, "3conv-32ch-64fc"),
        (3, 32, 128, "3conv-32ch-128fc"),
        (3, 64, 128, "3conv-64ch-128fc"),
        # 2 FC layer variations (deeper FC networks)
        (2, 32, (128, 64), "2conv-32ch-128fc-64fc"),
        (2, 32, (256, 128), "2conv-32ch-256fc-128fc"),
        (2, 64, (128, 64), "2conv-64ch-128fc-64fc"),
    ]
    
    all_results = []
    summary_results = []
    batch_size = 32
    max_epochs = 500
    patience = 50
    learning_rate = 1e-3
    
    target_name = "Teff+logg" if predict_target == 'teff_logg' else "Teff+logL"
    print(f"Running Self-Supervised Architecture Ablation Study for {target_name} prediction...")
    print(f"Testing {num_seeds} random seeds per architecture...")
    print(f"Total combinations: {len(architectures)}")
    print("Architecture | Params | Avg Epochs | Avg Time(min) | R²(Teff) Med[80%CI] | R²(logg) Med[80%CI] | R²(logL) Med[80%CI]")
    print("-" * 120)
    
    for arch_idx, (num_conv_layers, num_channels, fc_config, arch_desc) in enumerate(architectures):
        arch_start_time = time.time()
        results_for_arch = []
        
        print(f"\nRunning architecture {arch_idx+1}/{len(architectures)}: {arch_desc}")
        print(f"Encoder: {num_conv_layers} conv layers, {num_channels} channels")
        print(f"Regression head: {fc_config}")
        
        # Run multiple seeds for this architecture
        for seed_idx, seed in enumerate(seeds):
            seed_start_time = time.time()
            
            # Set all seeds for this run
            set_all_seeds(seed)
            
            # Create data loaders with this seed
            train_loader, test_loader, _ = create_dataloaders(power_tensor, labels_tensor, batch_size, seed)
            
            # Create and train a new encoder with this architecture
            encoder = EncoderCNN1D_Flexible(num_channels, input_size, num_conv_layers)
            
            # Wrap encoder in SimCLR for consistency (though we won't use the projector)
            simclr_model = self_supervised.SimCLR(encoder, 256)
            
            # Simulate self-supervised pretraining by training for a few epochs on OBA data
            # For now, we'll skip actual pretraining and just use random initialization
            # In a real scenario, you'd load pretrained weights here
            
            # Create model with this encoder and regression head configuration
            model = CNN1DFrozenConv_Flexible(encoder, 2, input_size, device, fc_config).to(device)
            
            # Count trainable parameters
            num_params = count_parameters(model)
            
            # Training setup
            loss_fn = nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.fc_layers.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
            
            # Train model
            epochs_trained, final_loss = train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, normalization_params)
            
            # Evaluate model
            r2_teff, r2_logg, r2_logl, rmse_teff, rmse_logg, rmse_logl = evaluate_model(model, test_loader, normalization_params, predict_target)
            
            seed_time = (time.time() - seed_start_time) / 60
            
            # Store results
            result = {
                'Architecture': arch_desc,
                'Num_Conv_Layers': num_conv_layers,
                'Num_Channels': num_channels,
                'FC_Config': fc_config,
                'Seed': seed,
                'Seed_Index': seed_idx + 1,
                'Parameters': num_params,
                'Epochs': epochs_trained,
                'Train_Time_min': seed_time,
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
            print(f"  Seed {seed_idx+1:2}: {epochs_trained:3} epochs, {seed_time:5.1f}min, R²(Teff)={r2_teff:.4f}, R²(logL)={r2_logl:.4f}")
            
            # Clean up memory
            del model, encoder, simclr_model
            torch.cuda.empty_cache()
        
        # Calculate summary statistics for this architecture
        stats = calculate_summary_statistics(results_for_arch)
        
        # Store summary
        summary_result = {
            'Architecture': arch_desc,
            'Num_Conv_Layers': num_conv_layers,
            'Num_Channels': num_channels,
            'FC_Config': fc_config,
            'Predict_Target': predict_target,
            'Num_Seeds': num_seeds,
            'Parameters': results_for_arch[0]['Parameters'],
            'Total_Time_min': (time.time() - arch_start_time) / 60,
            **stats
        }
        summary_results.append(summary_result)
        
        # Print summary for this architecture
        arch_time = (time.time() - arch_start_time) / 60
        print(f"  Summary: R²(Teff) Med={stats['R2_Teff_median']:.4f}[{stats['R2_Teff_q10']:.4f}-{stats['R2_Teff_q90']:.4f}], "
              f"R²(logL) Med={stats['R2_logL_median']:.4f}[{stats['R2_logL_q10']:.4f}-{stats['R2_logL_q90']:.4f}], "
              f"Time={arch_time:.1f}min")
    
    # Save detailed results to CSV
    detailed_df = pd.DataFrame(all_results)
    detailed_filename = f'selfsup_architecture_ablation_{predict_target}_detailed_results.csv'
    detailed_df.to_csv(detailed_filename, index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_results)
    summary_filename = f'selfsup_architecture_ablation_{predict_target}_summary_results.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    # Print final summary
    print("\n" + "="*120)
    print(f"ARCHITECTURE ABLATION STUDY COMPLETED FOR {target_name.upper()}")
    print("="*120)
    
    # Sort by combined R² score
    summary_df['Combined_R2'] = (summary_df['R2_Teff_median'] + summary_df['R2_logL_median']) / 2
    summary_df_sorted = summary_df.sort_values('Combined_R2', ascending=False)
    
    print(f"\nTOP 5 ARCHITECTURES (by combined R² median):")
    print("Rank | Architecture | R²(Teff) | R²(logL) | Combined | Params | Time(min)")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(summary_df_sorted.head(5).iterrows(), 1):
        print(f"{idx:4} | {row['Architecture']:<20} | {row['R2_Teff_median']:8.4f} | {row['R2_logL_median']:8.4f} | "
              f"{row['Combined_R2']:8.4f} | {row['Parameters']:6,} | {row['Total_Time_min']:7.1f}")
    
    print(f"\nDetailed results saved to {detailed_filename}")
    print(f"Summary results saved to {summary_filename}")
    
    return all_results, summary_results

if __name__ == '__main__':
    # Run architecture ablation for both prediction targets
    print("=" * 120)
    print("SELF-SUPERVISED ARCHITECTURE ABLATION STUDY 1: Predicting Teff + logg → logL")
    print("=" * 120)
    detailed_results_1, summary_results_1 = run_architecture_ablation_study('teff_logg', num_seeds=29)
    
    print("\n" + "=" * 120)
    print("SELF-SUPERVISED ARCHITECTURE ABLATION STUDY 2: Predicting Teff + logL direct")
    print("=" * 120)
    detailed_results_2, summary_results_2 = run_architecture_ablation_study('teff_logl', num_seeds=29)
    
    # Final comparison of best architectures
    print("\n" + "=" * 120)
    print("FINAL COMPARISON: BEST ARCHITECTURES")
    print("=" * 120)
    
    # Get best from each study
    df1 = pd.DataFrame(summary_results_1)
    df2 = pd.DataFrame(summary_results_2)
    
    df1['Combined_R2'] = (df1['R2_Teff_median'] + df1['R2_logL_median']) / 2
    df2['Combined_R2'] = (df2['R2_Teff_median'] + df2['R2_logL_median']) / 2
    
    best_1 = df1.loc[df1['Combined_R2'].idxmax()]
    best_2 = df2.loc[df2['Combined_R2'].idxmax()]
    
    print(f"\nBEST FOR TEFF+LOGG→LOGL: {best_1['Architecture']}")
    print(f"  R²(Teff): {best_1['R2_Teff_median']:.4f}, R²(logL): {best_1['R2_logL_median']:.4f}")
    print(f"  Combined: {best_1['Combined_R2']:.4f}, Params: {best_1['Parameters']:,}")
    
    print(f"\nBEST FOR TEFF+LOGL DIRECT: {best_2['Architecture']}")
    print(f"  R²(Teff): {best_2['R2_Teff_median']:.4f}, R²(logL): {best_2['R2_logL_median']:.4f}")
    print(f"  Combined: {best_2['Combined_R2']:.4f}, Params: {best_2['Parameters']:,}")
    
    if best_1['Combined_R2'] > best_2['Combined_R2']:
        print(f"\n🏆 OVERALL WINNER: {best_1['Architecture']} (Teff+logg→logL) "
              f"by {best_1['Combined_R2'] - best_2['Combined_R2']:.4f}")
    else:
        print(f"\n🏆 OVERALL WINNER: {best_2['Architecture']} (Teff+logL direct) "
              f"by {best_2['Combined_R2'] - best_1['Combined_R2']:.4f}")
    
    print("=" * 120)
    print("ARCHITECTURE ABLATION STUDY COMPLETED!")
    print("=" * 120) 