import pandas as pd
import numpy as np

def load_and_analyze_results():
    """
    Load results from CNN1 and MLP2 ablation studies and rank seeds by combined performance
    Only looking at the best architectures:
    - CNN1: 3conv-64ch-128fc (predicting Teff+logg, but we only care about Teff and derived logL)
    - MLP2: 3layer-512-256-128 (predicting Teff+logL directly)
    """
    
    # Load detailed results from CNN1 and MLP2 studies
    cnn1_all_results = pd.read_csv('ablation_study_cnn_teff_logg_detailed_results.csv')  # CNN1
    mlp2_all_results = pd.read_csv('ablation_study_mlp_teff_logl_detailed_results.csv')  # MLP2
    
    # Filter for only the best architectures
    cnn1_best_arch = '3conv-64ch-128fc'
    mlp2_best_arch = '3layer-512-256-128'
    
    cnn1_results = cnn1_all_results[cnn1_all_results['Architecture'] == cnn1_best_arch].copy()
    mlp2_results = mlp2_all_results[mlp2_all_results['Architecture'] == mlp2_best_arch].copy()
    
    print("Files loaded successfully!")
    print(f"CNN1 (teff_logg) with {cnn1_best_arch}: {len(cnn1_results)} results")
    print(f"MLP2 (teff_logl) with {mlp2_best_arch}: {len(mlp2_results)} results")
    
    # Verify we have the expected number of seeds
    cnn1_seeds = set(cnn1_results['Seed'].unique())
    mlp2_seeds = set(mlp2_results['Seed'].unique())
    
    if len(cnn1_seeds) != 29 or len(mlp2_seeds) != 29:
        print(f"WARNING: Expected 29 seeds for each architecture")
        print(f"CNN1 has {len(cnn1_seeds)} seeds, MLP2 has {len(mlp2_seeds)} seeds")
    
    # Get common seeds (should be the same 29 seeds)
    common_seeds = sorted(cnn1_seeds.intersection(mlp2_seeds))
    print(f"\nNumber of common seeds: {len(common_seeds)}")
    print(f"Seeds: {common_seeds}")
    
    # Function to calculate combined R² score (Teff + logL) / 2
    def calculate_combined_r2(df):
        return (df['R2_Teff'] + df['R2_logL']) / 2
    
    # Add combined R² scores to both dataframes
    cnn1_results['Combined_R2'] = calculate_combined_r2(cnn1_results)
    mlp2_results['Combined_R2'] = calculate_combined_r2(mlp2_results)
    
    # For each seed, get the performance with the best architectures
    seed_results = []
    
    for seed in common_seeds:
        # Get CNN1 performance for this seed with 3conv-64ch-128fc
        cnn1_seed_row = cnn1_results[cnn1_results['Seed'] == seed]
        if len(cnn1_seed_row) == 0:
            print(f"WARNING: No CNN1 data for seed {seed}")
            continue
        cnn1_row = cnn1_seed_row.iloc[0]  # Should be only one row per seed
        
        cnn1_combined = cnn1_row['Combined_R2']
        cnn1_teff = cnn1_row['R2_Teff']
        cnn1_logl = cnn1_row['R2_logL']
        
        # Get MLP2 performance for this seed with 3layer-512-256-128
        mlp2_seed_row = mlp2_results[mlp2_results['Seed'] == seed]
        if len(mlp2_seed_row) == 0:
            print(f"WARNING: No MLP2 data for seed {seed}")
            continue
        mlp2_row = mlp2_seed_row.iloc[0]  # Should be only one row per seed
        
        mlp2_combined = mlp2_row['Combined_R2']
        mlp2_teff = mlp2_row['R2_Teff']
        mlp2_logl = mlp2_row['R2_logL']
        
        # Calculate combined performance across both studies (average of CNN1 and MLP2 combined scores)
        combined_performance = (cnn1_combined + mlp2_combined) / 2
        
        seed_results.append({
            'Seed': seed,
            'CNN1_Combined_R2': cnn1_combined,
            'CNN1_R2_Teff': cnn1_teff,
            'CNN1_R2_logL': cnn1_logl,
            'MLP2_Combined_R2': mlp2_combined,
            'MLP2_R2_Teff': mlp2_teff,
            'MLP2_R2_logL': mlp2_logl,
            'Combined_Performance': combined_performance
        })
    
    # Convert to DataFrame and sort by combined performance
    seed_df = pd.DataFrame(seed_results)
    seed_df_sorted = seed_df.sort_values('Combined_Performance', ascending=False).reset_index(drop=True)
    
    # Add ranking
    seed_df_sorted['Rank'] = range(1, len(seed_df_sorted) + 1)
    
    # Display full ranking
    print("\n" + "="*120)
    print("FULL SEED RANKING BY COMBINED PERFORMANCE")
    print(f"CNN1: {cnn1_best_arch} | MLP2: {mlp2_best_arch}")
    print("="*120)
    print(f"{'Rank':<4} {'Seed':<8} {'Combined':<10} {'CNN1_Comb':<10} {'MLP2_Comb':<10} {'CNN1_Teff':<10} {'CNN1_logL':<10} {'MLP2_Teff':<10} {'MLP2_logL':<10}")
    print("-"*120)
    
    for idx, row in seed_df_sorted.iterrows():
        print(f"{row['Rank']:<4} {row['Seed']:<8} {row['Combined_Performance']:<10.5f} {row['CNN1_Combined_R2']:<10.5f} {row['MLP2_Combined_R2']:<10.5f} "
              f"{row['CNN1_R2_Teff']:<10.5f} {row['CNN1_R2_logL']:<10.5f} {row['MLP2_R2_Teff']:<10.5f} {row['MLP2_R2_logL']:<10.5f}")
    
    # Find the median seed (15th place out of 29)
    median_rank = 15
    median_seed_info = seed_df_sorted[seed_df_sorted['Rank'] == median_rank].iloc[0]
    
    print(f"\n" + "="*80)
    print(f"MEDIAN SEED ANALYSIS (RANK {median_rank} OUT OF {len(seed_df_sorted)})")
    print("="*80)
    print(f"Median Seed: {median_seed_info['Seed']}")
    print(f"Combined Performance Score: {median_seed_info['Combined_Performance']:.5f}")
    print(f"")
    print(f"Performance with Best Architectures:")
    print(f"  CNN1 ({cnn1_best_arch}):")
    print(f"    Combined R² (Teff+logL): {median_seed_info['CNN1_Combined_R2']:.5f}")
    print(f"    R² (Teff): {median_seed_info['CNN1_R2_Teff']:.5f}")
    print(f"    R² (logL): {median_seed_info['CNN1_R2_logL']:.5f}")
    print(f"")
    print(f"  MLP2 ({mlp2_best_arch}):")
    print(f"    Combined R² (Teff+logL): {median_seed_info['MLP2_Combined_R2']:.5f}")
    print(f"    R² (Teff): {median_seed_info['MLP2_R2_Teff']:.5f}")
    print(f"    R² (logL): {median_seed_info['MLP2_R2_logL']:.5f}")
    
    # Performance statistics
    print(f"\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    print(f"Combined Performance Statistics:")
    print(f"  Best:   {seed_df_sorted['Combined_Performance'].max():.5f} (Seed: {seed_df_sorted.iloc[0]['Seed']})")
    print(f"  Median: {seed_df_sorted['Combined_Performance'].median():.5f}")
    print(f"  Mean:   {seed_df_sorted['Combined_Performance'].mean():.5f} ± {seed_df_sorted['Combined_Performance'].std():.5f}")
    print(f"  Worst:  {seed_df_sorted['Combined_Performance'].min():.5f} (Seed: {seed_df_sorted.iloc[-1]['Seed']})")
    print(f"  Range:  {seed_df_sorted['Combined_Performance'].max() - seed_df_sorted['Combined_Performance'].min():.5f}")
    
    print(f"\nCNN1 vs MLP2 Performance Comparison:")
    print(f"  CNN1 Combined R²: {seed_df_sorted['CNN1_Combined_R2'].mean():.5f} ± {seed_df_sorted['CNN1_Combined_R2'].std():.5f}")
    print(f"  MLP2 Combined R²: {seed_df_sorted['MLP2_Combined_R2'].mean():.5f} ± {seed_df_sorted['MLP2_Combined_R2'].std():.5f}")
    print(f"  Difference:  {seed_df_sorted['CNN1_Combined_R2'].mean() - seed_df_sorted['MLP2_Combined_R2'].mean():.5f}")
    
    # Individual metric comparisons
    print(f"\nIndividual Metric Comparisons:")
    print(f"  Teff R² - CNN1: {seed_df_sorted['CNN1_R2_Teff'].mean():.5f} ± {seed_df_sorted['CNN1_R2_Teff'].std():.5f}")
    print(f"  Teff R² - MLP2: {seed_df_sorted['MLP2_R2_Teff'].mean():.5f} ± {seed_df_sorted['MLP2_R2_Teff'].std():.5f}")
    print(f"  logL R² - CNN1: {seed_df_sorted['CNN1_R2_logL'].mean():.5f} ± {seed_df_sorted['CNN1_R2_logL'].std():.5f}")
    print(f"  logL R² - MLP2: {seed_df_sorted['MLP2_R2_logL'].mean():.5f} ± {seed_df_sorted['MLP2_R2_logL'].std():.5f}")
    
    # Save results to CSV
    output_filename = 'seed_ranking_best_architectures.csv'
    seed_df_sorted.to_csv(output_filename, index=False)
    print(f"\nResults saved to: {output_filename}")
    
    return seed_df_sorted, median_seed_info

if __name__ == '__main__':
    try:
        seed_rankings, median_seed = load_and_analyze_results()
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"The median seed (15th place out of {len(seed_rankings)}) is: {median_seed['Seed']}")
        print(f"Combined performance score: {median_seed['Combined_Performance']:.5f}")
        print(f"This seed performs consistently well with both best architectures.")
        print(f"Architectures: CNN1 (3conv-64ch-128fc) + MLP2 (3layer-512-256-128)")
        print(f"Focus: Average of (Teff + logL R²)/2 scores from both studies")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files.")
        print(f"Please make sure you have run both ablation studies first:")
        print(f"  - ablation_study_cnn.py (for CNN1)")
        print(f"  - ablation_study_mlp.py (for MLP2)")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}") 