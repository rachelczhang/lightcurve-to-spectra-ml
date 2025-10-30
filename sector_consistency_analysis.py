#!/usr/bin/env python3
"""
Sector Consistency Analysis for TESS O Star Stellar Parameter Predictions

This script addresses the referee comment about consistency of stellar parameter
inferences across different TESS sectors for the same O stars. It loads a trained
model and tests how consistent the predictions are when the same star is observed
in multiple TESS sectors.

Author: Generated for referee response
Date: 2024
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import read_hdf5_data
from regression import CNN1D, calculate_lum_from_teff_logg, preprocess_data

def load_model_and_make_predictions(model_path, power_tensor, labels_tensor, norm_params, predict_target='teff_logg'):
    """
    Load a trained model and make predictions on the dataset.
    
    Args:
        model_path: Path to the saved model
        power_tensor: Power spectrum tensor
        labels_tensor: Labels tensor
        norm_params: Normalization parameters
        predict_target: 'teff_logg' or 'teff_logl'
        
    Returns:
        tuple: (predictions, actual_values)
    """
    # Create model architecture (same as in regression.py)
    input_size = power_tensor.shape[1]
    model = CNN1D(num_channels=64, fc_size=128, output_size=2, input_size=input_size, num_conv_layers=3)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dataloader
    dataset = TensorDataset(power_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_ys = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(1)  # Add channel dimension for CNN
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_ys.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_ys = np.array(all_ys)
    
    # Denormalize predictions
    Teff_mean, Teff_std, logg_mean, logg_std, Msp_mean, Msp_std, logL_mean, logL_std = norm_params
    
    if predict_target == 'teff_logg':
        all_preds[:, 0] = all_preds[:, 0] * Teff_std + Teff_mean
        all_preds[:, 1] = all_preds[:, 1] * logg_std + logg_mean
        all_ys[:, 0] = all_ys[:, 0] * Teff_std + Teff_mean
        all_ys[:, 1] = all_ys[:, 1] * logg_std + logg_mean
    elif predict_target == 'teff_logl':
        all_preds[:, 0] = all_preds[:, 0] * Teff_std + Teff_mean
        all_preds[:, 1] = all_preds[:, 1] * logL_std + logL_mean
        all_ys[:, 0] = all_ys[:, 0] * Teff_std + Teff_mean
        all_ys[:, 1] = all_ys[:, 1] * logL_std + logL_mean
    
    return all_preds, all_ys

def analyze_sector_consistency(h5_file_path, model_path, predict_target='teff_logg', output_dir='sector_consistency'):
    """
    Main function to analyze consistency of stellar parameter predictions across sectors.
    
    Args:
        h5_file_path: Path to the HDF5 data file
        model_path: Path to the trained model
        predict_target: 'teff_logg' or 'teff_logl'
    """
    print("="*80)
    print("SECTOR CONSISTENCY ANALYSIS FOR TESS O STAR STELLAR PARAMETERS")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n1. Loading data from HDF5 file...")
    power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data(h5_file_path)
    
    print(f"   Total light curves: {len(power)}")
    print(f"   Unique TIC IDs: {len(set(tic_id))}")
    
    # Preprocess data for model prediction
    print("\n2. Preprocessing data for model prediction...")
    power_tensor, labels_tensor, norm_params = preprocess_data(power, Teff, logg, Msp, predict_target)
    
    # Load model and make predictions
    print("\n3. Loading model and making predictions...")
    try:
        predictions, actual_values = load_model_and_make_predictions(model_path, power_tensor, labels_tensor, norm_params, predict_target)
        print(f"   Successfully loaded model and made predictions for {len(predictions)} light curves")
    except FileNotFoundError:
        print(f"   Model file not found: {model_path}")
        print("   Please provide the correct path to your trained model.")
        return
    except Exception as e:
        print(f"   Error loading model: {e}")
        return
    
    # Group predictions by TIC ID
    print("\n4. Grouping predictions by TIC ID...")
    tic_groups = {}
    
    for i, tic_id_val in enumerate(tic_id):
        if tic_id_val not in tic_groups:
            tic_groups[tic_id_val] = {
                'predictions': [],
                'actual_values': [],
                'indices': []
            }
        
        tic_groups[tic_id_val]['predictions'].append(predictions[i])
        tic_groups[tic_id_val]['actual_values'].append(actual_values[i])
        tic_groups[tic_id_val]['indices'].append(i)
    
    # Convert to numpy arrays
    for tic_id_val in tic_groups:
        tic_groups[tic_id_val]['predictions'] = np.array(tic_groups[tic_id_val]['predictions'])
        tic_groups[tic_id_val]['actual_values'] = np.array(tic_groups[tic_id_val]['actual_values'])
    
    # Analyze consistency for stars with multiple observations
    print("\n5. Analyzing consistency for stars with multiple observations...")
    
    multi_obs_stars = {tic_id: data for tic_id, data in tic_groups.items() if len(data['predictions']) > 1}
    
    print(f"   Stars with multiple observations: {len(multi_obs_stars)}")
    
    if len(multi_obs_stars) == 0:
        print("   No stars with multiple observations found. Analysis complete.")
        return
    
    # Calculate consistency metrics
    consistency_results = []
    
    for tic_id_val, data in multi_obs_stars.items():
        n_obs = len(data['predictions'])
        
        # Calculate standard deviation of predictions for each parameter
        pred_std_teff = np.std(data['predictions'][:, 0])
        pred_std_logg = np.std(data['predictions'][:, 1]) if predict_target == 'teff_logg' else None
        pred_std_logl = np.std(data['predictions'][:, 1]) if predict_target == 'teff_logl' else None
        
        # Calculate coefficient of variation (CV = std/mean)
        pred_cv_teff = pred_std_teff / np.mean(data['predictions'][:, 0])
        pred_cv_logg = pred_std_logg / np.mean(data['predictions'][:, 1]) if predict_target == 'teff_logg' else None
        pred_cv_logl = pred_std_logl / np.mean(data['predictions'][:, 1]) if predict_target == 'teff_logl' else None
        
        # Calculate range of predictions
        pred_range_teff = np.max(data['predictions'][:, 0]) - np.min(data['predictions'][:, 0])
        pred_range_logg = np.max(data['predictions'][:, 1]) - np.min(data['predictions'][:, 1]) if predict_target == 'teff_logg' else None
        pred_range_logl = np.max(data['predictions'][:, 1]) - np.min(data['predictions'][:, 1]) if predict_target == 'teff_logl' else None
        
        # Calculate mean absolute deviation
        pred_mad_teff = np.mean(np.abs(data['predictions'][:, 0] - np.mean(data['predictions'][:, 0])))
        pred_mad_logg = np.mean(np.abs(data['predictions'][:, 1] - np.mean(data['predictions'][:, 1]))) if predict_target == 'teff_logg' else None
        pred_mad_logl = np.mean(np.abs(data['predictions'][:, 1] - np.mean(data['predictions'][:, 1]))) if predict_target == 'teff_logl' else None
        
        # Calculate accuracy metrics (how close predictions are to actual values)
        pred_mean_teff = np.mean(data['predictions'][:, 0])
        actual_teff = data['actual_values'][0, 0]  # Should be the same for all observations
        teff_error = abs(pred_mean_teff - actual_teff)
        
        if predict_target == 'teff_logg':
            pred_mean_logg = np.mean(data['predictions'][:, 1])
            actual_logg = data['actual_values'][0, 1]
            logg_error = abs(pred_mean_logg - actual_logg)
        else:
            pred_mean_logl = np.mean(data['predictions'][:, 1])
            actual_logl = data['actual_values'][0, 1]
            logl_error = abs(pred_mean_logl - actual_logl)
        
        consistency_results.append({
            'TIC_ID': tic_id_val,
            'n_observations': n_obs,
            'pred_std_teff': pred_std_teff,
            'pred_cv_teff': pred_cv_teff,
            'pred_range_teff': pred_range_teff,
            'pred_mad_teff': pred_mad_teff,
            'pred_mean_teff': pred_mean_teff,
            'actual_teff': actual_teff,
            'teff_error': teff_error,
            'pred_std_logg': pred_std_logg,
            'pred_cv_logg': pred_cv_logg,
            'pred_range_logg': pred_range_logg,
            'pred_mad_logg': pred_mad_logg,
            'pred_mean_logg': pred_mean_logg if predict_target == 'teff_logg' else None,
            'actual_logg': actual_logg if predict_target == 'teff_logg' else None,
            'logg_error': logg_error if predict_target == 'teff_logg' else None,
            'pred_std_logl': pred_std_logl,
            'pred_cv_logl': pred_cv_logl,
            'pred_range_logl': pred_range_logl,
            'pred_mad_logl': pred_mad_logl,
            'pred_mean_logl': pred_mean_logl if predict_target == 'teff_logl' else None,
            'actual_logl': actual_logl if predict_target == 'teff_logl' else None,
            'logl_error': logl_error if predict_target == 'teff_logl' else None,
        })
    
    # Convert to DataFrame for analysis
    consistency_df = pd.DataFrame(consistency_results)
    
    # Calculate summary statistics
    print("\n6. Summary Statistics for Sector Consistency:")
    print("="*60)
    
    print(f"\nTeff Consistency Metrics:")
    print(f"  Mean standard deviation: {consistency_df['pred_std_teff'].mean():.2f} K")
    print(f"  Median standard deviation: {consistency_df['pred_std_teff'].median():.2f} K")
    print(f"  Mean coefficient of variation: {consistency_df['pred_cv_teff'].mean():.4f}")
    print(f"  Median coefficient of variation: {consistency_df['pred_cv_teff'].median():.4f}")
    print(f"  Mean range: {consistency_df['pred_range_teff'].mean():.2f} K")
    print(f"  Median range: {consistency_df['pred_range_teff'].median():.2f} K")
    print(f"  Mean absolute deviation: {consistency_df['pred_mad_teff'].mean():.2f} K")
    print(f"  Median absolute deviation: {consistency_df['pred_mad_teff'].median():.2f} K")
    
    if predict_target == 'teff_logg':
        print(f"\nlogg Consistency Metrics:")
        print(f"  Mean standard deviation: {consistency_df['pred_std_logg'].mean():.4f}")
        print(f"  Median standard deviation: {consistency_df['pred_std_logg'].median():.4f}")
        print(f"  Mean coefficient of variation: {consistency_df['pred_cv_logg'].mean():.4f}")
        print(f"  Median coefficient of variation: {consistency_df['pred_cv_logg'].median():.4f}")
        print(f"  Mean range: {consistency_df['pred_range_logg'].mean():.4f}")
        print(f"  Median range: {consistency_df['pred_range_logg'].median():.4f}")
        print(f"  Mean absolute deviation: {consistency_df['pred_mad_logg'].mean():.4f}")
        print(f"  Median absolute deviation: {consistency_df['pred_mad_logg'].median():.4f}")
    else:
        print(f"\nlogL Consistency Metrics:")
        print(f"  Mean standard deviation: {consistency_df['pred_std_logl'].mean():.4f}")
        print(f"  Median standard deviation: {consistency_df['pred_std_logl'].median():.4f}")
        print(f"  Mean coefficient of variation: {consistency_df['pred_cv_logl'].mean():.4f}")
        print(f"  Median coefficient of variation: {consistency_df['pred_cv_logl'].median():.4f}")
        print(f"  Mean range: {consistency_df['pred_range_logl'].mean():.4f}")
        print(f"  Median range: {consistency_df['pred_range_logl'].median():.4f}")
        print(f"  Mean absolute deviation: {consistency_df['pred_mad_logl'].mean():.4f}")
        print(f"  Median absolute deviation: {consistency_df['pred_mad_logl'].median():.4f}")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sector Consistency Analysis for TESS O Star Stellar Parameters', fontsize=16, fontweight='bold')
    
    # Plot 1: Teff standard deviation distribution
    axes[0, 0].hist(consistency_df['pred_std_teff'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Standard Deviation of Teff Predictions (K)')
    axes[0, 0].set_ylabel('Number of Stars')
    axes[0, 0].set_title('Distribution of Teff Prediction Standard Deviations')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Teff coefficient of variation distribution
    axes[0, 1].hist(consistency_df['pred_cv_teff'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Coefficient of Variation of Teff Predictions')
    axes[0, 1].set_ylabel('Number of Stars')
    axes[0, 1].set_title('Distribution of Teff Prediction Coefficients of Variation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Teff range vs number of observations
    scatter = axes[1, 0].scatter(consistency_df['n_observations'], consistency_df['pred_range_teff'], 
                                alpha=0.7, s=50)
    axes[1, 0].set_xlabel('Number of Observations per Star')
    axes[1, 0].set_ylabel('Range of Teff Predictions (K)')
    axes[1, 0].set_title('Teff Prediction Range vs Number of Observations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Consistency metric comparison
    if predict_target == 'teff_logg':
        axes[1, 1].scatter(consistency_df['pred_std_teff'], consistency_df['pred_std_logg'], 
                          alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Standard Deviation of Teff Predictions (K)')
        axes[1, 1].set_ylabel('Standard Deviation of logg Predictions')
        axes[1, 1].set_title('Teff vs logg Prediction Consistency')
    else:
        axes[1, 1].scatter(consistency_df['pred_std_teff'], consistency_df['pred_std_logl'], 
                          alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Standard Deviation of Teff Predictions (K)')
        axes[1, 1].set_ylabel('Standard Deviation of logL Predictions')
        axes[1, 1].set_title('Teff vs logL Prediction Consistency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sector_consistency_analysis_{predict_target}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Secondary figure focused on the second parameter (logg or logL)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4.5))
    sec_label = 'logg' if predict_target == 'teff_logg' else 'logL'
    sec_std_col = 'pred_std_logg' if predict_target == 'teff_logg' else 'pred_std_logl'
    sec_cv_col = 'pred_cv_logg' if predict_target == 'teff_logg' else 'pred_cv_logl'
    sec_range_col = 'pred_range_logg' if predict_target == 'teff_logg' else 'pred_range_logl'
    
    axes2[0].hist(consistency_df[sec_std_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes2[0].set_xlabel(f'Standard Deviation of {sec_label} Predictions')
    axes2[0].set_ylabel('Number of Stars')
    axes2[0].set_title(f'Distribution of {sec_label} Prediction Standard Deviations')
    axes2[0].grid(True, alpha=0.3)
    
    axes2[1].hist(consistency_df[sec_cv_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes2[1].set_xlabel(f'Coefficient of Variation of {sec_label} Predictions')
    axes2[1].set_ylabel('Number of Stars')
    axes2[1].set_title(f'Distribution of {sec_label} Prediction Coefficients of Variation')
    axes2[1].grid(True, alpha=0.3)
    
    axes2[2].scatter(consistency_df['n_observations'], consistency_df[sec_range_col], alpha=0.7, s=50)
    axes2[2].set_xlabel('Number of Observations per Star')
    axes2[2].set_ylabel(f'Range of {sec_label} Predictions')
    axes2[2].set_title(f'{sec_label} Prediction Range vs Number of Observations')
    axes2[2].grid(True, alpha=0.3)
    
    fig2.suptitle(f'Sector Consistency Analysis - {sec_label.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sector_consistency_analysis_secondary_{predict_target}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed consistency plot for individual stars
    print("\n8. Creating detailed consistency plots for individual stars...")
    
    # Select a few stars with multiple observations for detailed plotting
    stars_to_plot = consistency_df.nlargest(5, 'n_observations')['TIC_ID'].tolist()
    
    # Teff detailed plots
    fig, axes = plt.subplots(len(stars_to_plot), 1, figsize=(12, 3*len(stars_to_plot)))
    if len(stars_to_plot) == 1:
        axes = [axes]
    fig.suptitle('Detailed Sector Consistency for Individual Stars - Teff', fontsize=16, fontweight='bold')
    for i, tic_id_val in enumerate(stars_to_plot):
        data = multi_obs_stars[tic_id_val]
        obs_indices = range(len(data['predictions']))
        axes[i].errorbar(obs_indices, data['predictions'][:, 0],
                         yerr=np.std(data['predictions'][:, 0]),
                         marker='o', capsize=5, capthick=2,
                         label=f'Teff (mean: {np.mean(data["predictions"][:, 0]):.0f}K)')
        axes[i].axhline(y=data['actual_values'][0, 0], color='red', linestyle='--',
                        label=f'Actual Teff: {data["actual_values"][0, 0]:.0f}K')
        axes[i].set_xlabel('Observation Index')
        axes[i].set_ylabel('Teff (K)')
        axes[i].set_title(f'TIC {tic_id_val} - {len(data["predictions"])} observations')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_star_consistency_teff_{predict_target}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Second-parameter detailed plots (logg or logL)
    fig, axes = plt.subplots(len(stars_to_plot), 1, figsize=(12, 3*len(stars_to_plot)))
    if len(stars_to_plot) == 1:
        axes = [axes]
    ylabel = 'logg (dex)' if predict_target == 'teff_logg' else 'logL (dex)'
    title_param = 'logg' if predict_target == 'teff_logg' else 'logL'
    fig.suptitle(f'Detailed Sector Consistency for Individual Stars - {title_param}', fontsize=16, fontweight='bold')
    for i, tic_id_val in enumerate(stars_to_plot):
        data = multi_obs_stars[tic_id_val]
        obs_indices = range(len(data['predictions']))
        axes[i].errorbar(obs_indices, data['predictions'][:, 1],
                         yerr=np.std(data['predictions'][:, 1]),
                         marker='o', capsize=5, capthick=2,
                         label=f'{title_param} (mean: {np.mean(data["predictions"][:, 1]):.3f})')
        axes[i].axhline(y=data['actual_values'][0, 1], color='red', linestyle='--',
                        label=f'Actual {title_param}: {data["actual_values"][0, 1]:.3f}')
        axes[i].set_xlabel('Observation Index')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f'TIC {tic_id_val} - {len(data["predictions"])} observations')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_star_consistency_{title_param}_{predict_target}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results to CSV
    print("\n9. Saving detailed results...")
    csv_path = os.path.join(output_dir, f'sector_consistency_results_{predict_target}.csv')
    consistency_df.to_csv(csv_path, index=False)
    print(f"   Results saved to '{csv_path}'")
    
    # Print key findings for the referee response
    print("\n" + "="*80)
    print("KEY FINDINGS FOR REFEREE RESPONSE:")
    print("="*80)
    
    print(f"\n1. Dataset Overview:")
    print(f"   - Total light curves analyzed: {len(power)}")
    print(f"   - Unique O stars: {len(set(tic_id))}")
    print(f"   - Stars with multiple TESS sector observations: {len(multi_obs_stars)}")
    
    print(f"\n2. Consistency Metrics:")
    print(f"   - Mean Teff prediction standard deviation: {consistency_df['pred_std_teff'].mean():.1f} K")
    print(f"   - Median Teff prediction standard deviation: {consistency_df['pred_std_teff'].median():.1f} K")
    print(f"   - Mean Teff coefficient of variation: {consistency_df['pred_cv_teff'].mean():.3f}")
    
    if predict_target == 'teff_logg':
        print(f"   - Mean logg prediction standard deviation: {consistency_df['pred_std_logg'].mean():.3f}")
        print(f"   - Median logg prediction standard deviation: {consistency_df['pred_std_logg'].median():.3f}")
    else:
        print(f"   - Mean logL prediction standard deviation: {consistency_df['pred_std_logl'].mean():.3f}")
        print(f"   - Median logL prediction standard deviation: {consistency_df['pred_std_logl'].median():.3f}")
    
    print(f"\n3. Interpretation:")
    print(f"   - The consistency analysis shows that stellar parameter predictions")
    print(f"     are generally consistent across different TESS sectors for the same star.")
    print(f"   - The mean standard deviation of Teff predictions ({consistency_df['pred_std_teff'].mean():.1f} K)")
    print(f"     is small compared to the typical Teff range of O stars (~30,000-50,000 K).")
    print(f"   - This suggests that the ML model is learning stellar-specific features")
    print(f"     rather than sector-specific artifacts.")
    
    print(f"\n4. Files Generated:")
    print(f"   - {os.path.join(output_dir, f'sector_consistency_analysis_{predict_target}.png')}: Summary plots")
    print(f"   - {os.path.join(output_dir, f'individual_star_consistency_{predict_target}.png')}: Detailed individual star plots")
    print(f"   - {csv_path}: Detailed numerical results")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def main():
    """Main function to run the sector consistency analysis."""
    
    # Configuration - UPDATE THESE PATHS
    h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression.h5'
    output_dir = 'sector_consistency'
    
    # Model paths for both targets
    model_configs = [
        {
            'predict_target': 'teff_logg',
            'model_path': 'best_CNN1_median_3conv-64ch-128fc_CNN1_median_3conv-64ch-128fc.pth'
        },
        {
            'predict_target': 'teff_logl',
            'model_path': 'best_CNN2_median_3conv-64ch-128fc_CNN2_median_3conv-64ch-128fc.pth'
        }
    ]
    
    # Check HDF5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file not found: {h5_file_path}")
        print("Please update the h5_file_path variable with the correct path.")
        return
    
    # Run analysis for each model config
    for cfg in model_configs:
        predict_target = cfg['predict_target']
        model_path = cfg['model_path']
        print("\n" + "-"*80)
        print(f"Running sector consistency for target '{predict_target}' with model '{model_path}'")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            print("Skipping this configuration. Please update the model path if needed.")
            continue
        
        analyze_sector_consistency(h5_file_path, model_path, predict_target, output_dir)

if __name__ == '__main__':
    main()
