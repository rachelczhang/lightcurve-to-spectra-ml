import h5py
import numpy as np
import pandas as pd

def get_filtered_tic_ids(h5_file_path):
    """
    Extracts TIC IDs from the tessOregression.h5 file using the same filtering logic as regression.py
    Filters out datasets with uncertain measurements (containing '>' or '<' in Teff, logg, Msp)
    Returns both unique TIC IDs and total number of light curves after filtering
    """
    tic_ids = set()
    total_light_curves = 0
    filtered_out_count = 0
    
    with h5py.File(h5_file_path, 'r') as h5f:
        # Iterate through all datasets in the file
        for dataset_name in h5f.keys():
            # Skip the Frequency dataset
            if dataset_name != 'Frequency':
                dataset = h5f[dataset_name]
                
                # Apply the same filtering logic as in data.py
                if not any('>' in str(dataset.attrs[attr]) or '<' in str(dataset.attrs[attr]) for attr in ['Teff', 'logg', 'Msp']):
                    # This dataset passes the filter
                    tic_id = dataset.attrs['TIC_ID']
                    tic_ids.add(tic_id)
                    total_light_curves += 1
                else:
                    # This dataset was filtered out
                    filtered_out_count += 1
    
    # Convert to sorted list for better readability
    tic_ids_list = sorted(list(tic_ids))
    return tic_ids_list, total_light_curves, filtered_out_count

if __name__ == '__main__':
    h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression.h5'
    output_csv_path = '/mnt/home/rzhang/project/lightcurve-to-spectra-ml/tic_ids.csv'
    
    print("Extracting TIC IDs from tessOregression.h5 (post-filtered dataset)...")
    tic_ids, total_light_curves, filtered_out_count = get_filtered_tic_ids(h5_file_path)
    
    print(f"\nTotal number of unique TIC IDs (after filtering): {len(tic_ids)}")
    print(f"Total number of light curves (after filtering): {total_light_curves}")
    print(f"Number of datasets filtered out: {filtered_out_count}")
    print(f"Total datasets in file: {total_light_curves + filtered_out_count}")
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({'TIC_ID': tic_ids})
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nTIC IDs saved to: {output_csv_path}")
    print("First 10 TIC IDs:")
    print(df.head(10))

