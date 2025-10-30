import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import time
import wandb
import self_supervised
from torch.optim.lr_scheduler import ReduceLROnPlateau
from regression import read_hdf5_data, preprocess_data, create_dataloaders, calculate_lum_from_teff_logg, denormalize_data
from sklearn.metrics import r2_score, mean_squared_error
import cnn_selfsup 
import run_cnn
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

class CNN1DFrozenEverything(nn.Module):
	def __init__(self, pretrained_encoder, nonpretrained_projector):
		super().__init__()
		self.encoder = pretrained_encoder
		# freeze the encoder
		for param in self.encoder.parameters():
			param.requires_grad = False

		self.flatten = nn.Flatten()
		self.projector = nonpretrained_projector
		for param in self.projector.parameters():
			param.requires_grad = False

	def forward(self, x):
		x = self.encoder(x) 
		x = self.flatten(x)
		x = self.projector(x)
		return x

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler, max_epochs, patience, norm_params):
    """Train model with early stopping - simplified version without wandb logging"""
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

def run_ablation_study(predict_target='teff_logg', pretrained_model_path='best_selfsup46_1conv.pth', num_seeds=29):
    """
    Run ablation study for self-supervised regression with multiple random seeds
    
    Args:
        predict_target: 'teff_logg' or 'teff_logl'
        pretrained_model_path: path to pretrained self-supervised model
        num_seeds: number of random seeds to test
    """
    # Load data
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')
    power_tensor, labels_tensor, normalization_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    input_size = len(power.iloc[0])
    
    # Define the same 30 seeds as ablation study for consistency
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627,
             303132, 333435, 363738, 394041, 424344, 454647, 484950, 515253, 545556, 575859,
             606162, 636465, 666768, 697071, 727374, 757677, 787980, 818283, 848586, 878889][:num_seeds]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Single "architecture" - frozen self-supervised encoder
    architecture_desc = f"selfsup_frozen_encoder_{predict_target}"
    
    all_results = []
    batch_size = 32
    max_epochs = 500     # Updated to match other ablation studies
    patience = 50        # Updated to match other ablation studies
    learning_rate = 1e-3  # Use same as original
    
    target_name = "Teff+logg" if predict_target == 'teff_logg' else "Teff+logL"
    print(f"Running Self-Supervised ablation study for {target_name} prediction...")
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Testing {num_seeds} random seeds...")
    print("Seed | Epochs | Time(min) | R²(Teff) | R²(logg) | R²(logL) | Loss")
    print("-" * 80)
    
    arch_start_time = time.time()
    
    # Run multiple seeds for this configuration
    for seed_idx, seed in enumerate(seeds):
        seed_start_time = time.time()
        
        # Set all seeds for this run
        set_all_seeds(seed)
        
        # Create data loaders with this seed
        train_loader, test_loader, _ = create_dataloaders(power_tensor, labels_tensor, batch_size, seed)
        
        # Create pretrained model and extract encoder
        pretrained_model = self_supervised.SimCLR(self_supervised.EncoderCNN1D(32, input_size), 256)
        pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        pretrained_model.to(device)
        
        # Create model with frozen encoder
        model = cnn_selfsup.CNN1DFrozenConv(pretrained_model.encoder, 2, input_size, device).to(device)
        
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
            'Architecture': architecture_desc,
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
            'Predict_Target': predict_target,
            'Pretrained_Model': pretrained_model_path
        }
        all_results.append(result)
        
        # Print progress for this seed
        print(f"{seed_idx+1:4} | {epochs_trained:6} | {seed_time:8.1f} | {r2_teff:8.4f} | {r2_logg:8.4f} | {r2_logl:8.4f} | {final_loss:8.4f}")
        
        # Clean up memory
        del model, pretrained_model
        torch.cuda.empty_cache()
    
    # Calculate summary statistics
    stats = calculate_summary_statistics(all_results)
    
    # Print summary
    total_time = time.time() - arch_start_time
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS FOR {target_name.upper()} PREDICTION")
    print("="*80)
    print(f"Architecture: {architecture_desc}")
    print(f"Pretrained Model: {pretrained_model_path}")
    print(f"Number of Seeds: {num_seeds}")
    print(f"Trainable Parameters: {all_results[0]['Parameters']:,}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Average Time per Seed: {total_time/(60*num_seeds):.1f} minutes")
    print(f"Average Epochs: {stats['avg_epochs']:.1f}")
    
    print(f"\nR² SCORES:")
    print(f"  Teff  - Median: {stats['R2_Teff_median']:.4f} [80% CI: {stats['R2_Teff_q10']:.4f}-{stats['R2_Teff_q90']:.4f}]")
    if predict_target == 'teff_logg':
        print(f"  logg  - Median: {stats['R2_logg_median']:.4f} [80% CI: {stats['R2_logg_q10']:.4f}-{stats['R2_logg_q90']:.4f}]")
    print(f"  logL  - Median: {stats['R2_logL_median']:.4f} [80% CI: {stats['R2_logL_q10']:.4f}-{stats['R2_logL_q90']:.4f}]")
    
    print(f"\nRMSE SCORES (Median):")
    print(f"  Teff: {stats['RMSE_Teff_median']:.4f}")
    if predict_target == 'teff_logg':
        print(f"  logg: {stats['RMSE_logg_median']:.4f}")
    print(f"  logL: {stats['RMSE_logL_median']:.4f}")
    
    # Save detailed results to CSV
    detailed_df = pd.DataFrame(all_results)
    detailed_filename = f'selfsup_ablation_{predict_target}_detailed_results.csv'
    detailed_df.to_csv(detailed_filename, index=False)
    
    # Save summary statistics
    summary_result = {
        'Architecture': architecture_desc,
        'Predict_Target': predict_target,
        'Pretrained_Model': pretrained_model_path,
        'Num_Seeds': num_seeds,
        'Parameters': all_results[0]['Parameters'],
        'Total_Time_min': total_time / 60,
        **stats
    }
    summary_df = pd.DataFrame([summary_result])
    summary_filename = f'selfsup_ablation_{predict_target}_summary_results.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    print(f"\nDetailed results saved to {detailed_filename}")
    print(f"Summary results saved to {summary_filename}")
    
    return all_results, summary_result

if __name__ == '__main__':
    # Run ablation for both prediction targets with 29 seeds each
    print("=" * 80)
    print("SELF-SUPERVISED ABLATION STUDY 1: Pretrained1 (Teff + logg → logL)")
    print("=" * 80)
    detailed_results_1, summary_results_1 = run_ablation_study('teff_logg', num_seeds=29)
    
    print("\n" + "=" * 80)
    print("SELF-SUPERVISED ABLATION STUDY 2: Pretrained2 (Teff + logL direct)")
    print("=" * 80)
    detailed_results_2, summary_results_2 = run_ablation_study('teff_logl', num_seeds=29)
    
    # Final comparison
    print("\n" + "=" * 100)
    print("FINAL COMPARISON: PRETRAINED1 vs PRETRAINED2")
    print("=" * 100)
    
    print(f"\nPRETRAINED1 (Teff+logg → logL):")
    print(f"  R² Teff  - Median: {summary_results_1['R2_Teff_median']:.4f} [80% CI: {summary_results_1['R2_Teff_q10']:.4f}-{summary_results_1['R2_Teff_q90']:.4f}]")
    print(f"  R² logg  - Median: {summary_results_1['R2_logg_median']:.4f} [80% CI: {summary_results_1['R2_logg_q10']:.4f}-{summary_results_1['R2_logg_q90']:.4f}]")
    print(f"  R² logL  - Median: {summary_results_1['R2_logL_median']:.4f} [80% CI: {summary_results_1['R2_logL_q10']:.4f}-{summary_results_1['R2_logL_q90']:.4f}]")
    
    print(f"\nPRETRAINED2 (Teff+logL direct):")
    print(f"  R² Teff  - Median: {summary_results_2['R2_Teff_median']:.4f} [80% CI: {summary_results_2['R2_Teff_q10']:.4f}-{summary_results_2['R2_Teff_q90']:.4f}]")
    print(f"  R² logL  - Median: {summary_results_2['R2_logL_median']:.4f} [80% CI: {summary_results_2['R2_logL_q10']:.4f}-{summary_results_2['R2_logL_q90']:.4f}]")
    
    # Combined score comparison
    combined_1 = (summary_results_1['R2_Teff_median'] + summary_results_1['R2_logL_median']) / 2
    combined_2 = (summary_results_2['R2_Teff_median'] + summary_results_2['R2_logL_median']) / 2
    
    print(f"\nCOMBINED SCORE (Teff + logL R² median):")
    print(f"  Pretrained1: {combined_1:.4f}")
    print(f"  Pretrained2: {combined_2:.4f}")
    
    if combined_1 > combined_2:
        print(f"\n🏆 WINNER: Pretrained1 (Teff+logg → logL) by {combined_1-combined_2:.4f}")
    else:
        print(f"\n🏆 WINNER: Pretrained2 (Teff+logL direct) by {combined_2-combined_1:.4f}")
    
    print("=" * 100)
    print("SELF-SUPERVISED ABLATION STUDY COMPLETED!")
    print("=" * 100)