import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from run_mlp import load_data
import wandb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from data import read_hdf5_data
torch.manual_seed(42)
np.random.seed(42)

def power_spectrum_model(nu, alpha0, nu_char, gamma, Cw):
    result = alpha0 / (1 + (nu / nu_char)**gamma) + Cw
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print(f"Invalid value encountered in power_spectrum_model with parameters: alpha0={alpha0}, nu_char={nu_char}, gamma={gamma}, Cw={Cw}")
        print(f"nu: {nu}")
        print(f"result: {result}")
    return result 

# def find_oscillation_modes(power_spectrum):
#     noise_estimate = np.std(power_spectrum[-len(power_spectrum)//5:])
#     print('noise estimate', noise_estimate)
#     height_threshold = 4 * noise_estimate
#     print('height threshold', height_threshold)
#     prominence_threshold = noise_estimate
#     print('prominence threshold', prominence_threshold)
#     peaks, _ = find_peaks(power_spectrum, height=height_threshold, prominence=prominence_threshold)
#     if len(peaks) > 0:
#         return True
#     else:
#         return False

# def find_sharp_peaks(power_spectrum, factor=10, window=10):
#     for i in range(len(power_spectrum) - window):
#         if np.any(power_spectrum[i + 1:i + 1 + window] >= factor * power_spectrum[i]):
#             return True
#         else:
#             return False

if __name__ == '__main__':
    # # classification
    # benchmark_df = pd.DataFrame(columns=['alpha0', 'nu_char', 'gamma', 'Cw', 'labels'])
    # regression
    benchmark_df = pd.DataFrame(columns=['alpha0', 'nu_char', 'gamma', 'Cw', 'Teff', 'logg', 'Msp'])
    # # classification
    # power, logpower, labels, freq = load_data('tessOBAstars.h5')
    # regression
    power, Teff, logg, Msp, freq, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression_magunits.h5')  
    print('power', power)
    print('freq', freq)
    for i in range(len(power)):
        print('i', i)
        # print('label', labels[i])
        params, params_covariance = curve_fit(power_spectrum_model, freq[i], power[i], p0=[0.0003, 2, 3, 2e-5], bounds=([1e-7, 5e-3, 0, 0], [0.01, 12, 13, 0.1]))
        print('params', params)
        alpha0, nu_char, gamma, Cw = params
        print("Fitted parameters:")
        print(f"alpha0 (Amplitude): {alpha0}")
        print(f"nu_char (Characteristic Frequency): {nu_char}")
        print(f"gamma (Shape factor): {gamma}")
        print(f"Cw (Constant offset): {Cw}")
        # if gamma <= 4.5:
        benchmark_df = benchmark_df._append({'alpha0': alpha0, 'nu_char': nu_char, 'gamma': gamma, 'Cw': Cw}, ignore_index=True)
    # # classification
    # benchmark_df['labels'] = labels
    # regression
    benchmark_df['Teff'] = Teff
    benchmark_df['logg'] = logg
    benchmark_df['Msp'] = Msp
    print('benchmark df', benchmark_df)
    
    # plt.figure(figsize=(10, 6))
    # plt.scatter(freq[i], power[i], label='Data')
    # plt.plot(freq[i], power_spectrum_model(freq[i], *params), label='Fitted function', color='red')
    # plt.xlabel('Frequency (nu)')
    # plt.ylabel('Power Spectrum (alpha_nu)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('curve_fitting.png')
    # plt.clf()

        # if find_oscillation_modes(power[i]):
        #     params, params_covariance = curve_fit(power_spectrum_model, freq[i], power[i], p0=[0.0003, 2, 3, 2e-5], bounds=([1e-7, 5e-3, 0, 0], [0.01, 12, 13, 0.1]))
        #     print('params', params)
        #     alpha0, nu_char, gamma, Cw = params
        #     print("Fitted parameters:")
        #     print(f"alpha0 (Amplitude): {alpha0}")
        #     print(f"nu_char (Characteristic Frequency): {nu_char}")
        #     print(f"gamma (Shape factor): {gamma}")
        #     print(f"Cw (Constant offset): {Cw}")
        #     benchmark_df = benchmark_df._append({'alpha0': alpha0, 'nu_char': nu_char, 'gamma': gamma, 'Cw': Cw}, ignore_index=True)
        # else:
        #     print('Oscillation modes found')
        #     if i == 6:
        #         plt.figure(figsize=(10, 6))
        #         plt.scatter(freq[i], power[i], label='Data')
        #         # plt.plot(freq[i], power_spectrum_model(freq[i], *params), label='Fitted function', color='red')
        #         plt.xlabel('Frequency (nu)')
        #         plt.ylabel('Power Spectrum (alpha_nu)')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.legend()
        #         plt.savefig('curve_fitting.png')
        #         plt.clf()
    
    # # classification
    # benchmark_df.to_hdf('curvefitparams.h5', key='df', mode='w')
    # regression
    benchmark_df.to_hdf('curvefitparams_reg.h5', key='df', mode='w')