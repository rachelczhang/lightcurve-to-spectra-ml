import h5py
import numpy as np 
import pandas as pd 

def read_hdf5_data(hdf5_path):
    power = []
    Teff = []
    logg = []
    Msp = []
    tic_id = []
    with h5py.File(hdf5_path, 'r') as h5f:
        if 'Frequency' in h5f:
            frequencies = np.array(h5f['Frequency'])
        for name in h5f:
            if name != 'Frequency': 
                dataset = h5f[name]
                if not any('>' in str(dataset.attrs[attr]) or '<' in str(dataset.attrs[attr]) for attr in ['Teff', 'logg', 'Msp']):
                    power.append(list(dataset))
                    Teff.append(dataset.attrs['Teff'])
                    logg.append(dataset.attrs['logg'])
                    Msp.append(dataset.attrs['Msp'])
                    tic_id.append(dataset.attrs['TIC_ID'])
    power = pd.Series(power)
    return power, Teff, logg, Msp, frequencies, tic_id