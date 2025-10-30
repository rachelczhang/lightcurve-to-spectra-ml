import h5py
import pandas as pd 

h5_file_path = 'tessOBAstars.h5'
#h5_file_path = '/mnt/sdceph/users/rzhang/tessOBAstars_all.h5'

#with h5py.File(h5_file_path, 'r') as f:
    # Filter out the dataset names that follow the pattern 'TIC_*_Power'
#    dataset_names = [name for name in f.keys() if name.startswith('TIC_') and name.endswith('_Power')]
#    num_datasets = len(dataset_names)
#    print(f"Number of power spectra datasets: {num_datasets}")

with h5py.File(h5_file_path, 'r') as f:
    # Initialize a counter
    num_datasets = 0
    
    # Iterate through all items at the root level
    for key in f:
        # Check if the key matches the pattern directly
        if key.startswith('TIC_') and key.endswith('_Power'):
            num_datasets += 1
    
    print(f"Number of power spectra datasets: {num_datasets}")

with pd.HDFStore(h5_file_path, 'r') as store:
    df = store['df']
    num_rows = len(df)
    print(f"Number of rows in the DataFrame: {num_rows}")
