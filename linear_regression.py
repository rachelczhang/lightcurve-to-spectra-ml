# data = """
# 218.23589022037584, 8.40874500667088
# 214.88928710080236, 23.205524975343497
# 275.1676123445018, 237.87241732935018
# 346.95124989255584, 152.33359765771812
# 408.0733704412147, 256.21358339713146
# 408.0733704412147, 312.337159180088
# 408.0733704412147, 883.561042902456
# 577.7566679831825, 26.2636352765333
# 551.582829389114, 10.50764462712928
# 609.8666701068876, 26.2636352765333
# 890.562552734677, 6.247394411085379
# 977.0862143786246, 86.19535664753033
# 1149.2187010036987, 58.00171836601337
# 1176.1691896702687, 34.48533177622105
# 1449.018150486198, 32.81927872511474
# 1449.018150486198, 6.729100107235915
# 1203.7516980200671, 200.0208501442907
# 1577.5625109334908, 131.3044877950839
# 2035.751822261104, 131.3044877950839
# 1665.23875663253, 60.94614443529066
# 1898.9935512689922, 55.199543212815676
# 1898.9935512689922, 46.41588833612778
# 2035.751822261104, 30.469895709035082
# 3895.8661446875562, 20.00208501442905
# 3806.5971030386117, 56.583286926108485
# 5027.3865842818595, 53.84963893812914
# 3987.2286471313264, 102.50680283341828
# 4176.431561883492, 156.1523006000498
# 3987.2286471313264, 220.844211988575
# 5027.3865842818595, 262.63635276533324
# 5515.828293891133, 141.4287276844031
# 5733.096629695021, 256.21358339713146
# 4374.6125779984795, 1523.3359765771813
# 6588.581861506808, 1160.1553017399715
# 7398.2246853688675, 1131.783715491855
# 7063.066392045383, 800.2502278161053
# 7063.066392045383, 499.94788007281375
# 7398.2246853688675, 344.85331776221085
# 7931.016603333043, 362.3596111531805
# 8568.127214899465, 410.11270705513044
# 8701.563933188898, 640.4004271197283
# 7571.7214883373845, 249.94788278930028
# 7063.066392045383, 190.35745615898014
# 8905.625527346761, 185.70226648110975
# 9770.862143786237, 336.4199333410339
# 10637.648543163097, 172.40868515129355
# 11403.731409117325, 328.19278725114776
# 8502.178162653106, 1600.667308959397
# 9328.217371603681, 1903.5745615898015
# 10234.511400162823, 1249.6091412919868
# 11492.187010036974, 1160.1553017399715
# 12608.72407680677, 97.55450100468744
# 14490.18150486195, 243.8354098268829
# 11228.857500568243, 2499.4788278930055
# 10971.561867027238, 3807.546021222376
# 13105.381396580902, 4417.344703140074
# 13833.739627296194, 5384.963893812919
# 14158.156592244739, 4000.8340492444804
# 14158.156592244739, 2154.4346900318865
# 13516.75628313439, 1280.934378652548
# 14158.156592244739, 820.3109232014356
# 14715.846019280558, 840.8745006670888
# 14378.650278357578, 689.7785379387658
# 14158.156592244739, 487.7216596885466
# 16652.387566325284, 1857.0226648110972
# 18989.93551268992, 1903.5745615898015
# 19435.271149298263, 3364.1993334103395
# 21323.52798169755, 3534.9811050301096
# """
# with open('lum_vs_alpha.dat', 'w') as file:
#     file.write(data.strip())

# data = """
# 219.88352520875503, 1.022906799733014
# 214.89480346928372, 1.5611427731269973
# 276.60527949179067, 2.07985369888579
# 347.95913231831975, 2.5888968917971775
# 408.59747740624715, 1.4696472784789474
# 408.59747740624715, 1.2732627472046125
# 408.59747740624715, 1.1542372394900324
# 408.59747740624715, 0.9702535622289161
# 408.59747740624715, 0.8663730174238895
# 550.6349675300588, 0.7853838512085994
# 576.4973856945102, 0.8280046712286473
# 603.5745190745841, 1.226096323444954
# 891.591897210636, 0.7736144803177563
# 977.3119803553491, 0.8033744818017781
# 1147.6267547614975, 1.2077226551344629
# 1201.5288946155802, 1.3627815006174744
# 1174.2685834509275, 1.633483507981373
# 1443.6716845961394, 1.2636864951346356
# 1443.6716845961394, 1.0306584164481178
# 1594.6222049631665, 1.9727950393776206
# 2052.5434471508693, 1.829342743352861
# 1915.9846552588526, 1.0865896745227324
# 1901.3837040719795, 0.9343117342198284
# 1669.518985427485, 0.6651978236777661
# 2052.5434471508693, 4.260967255066383
# 3993.2720981209945, 2.346884967986971
# 3993.2720981209945, 1.3024291219623356
# 4180.829516258267, 1.119902793272475
# 4377.196197634795, 0.6651978236777661
# 3902.672662312389, 4.04163767300011
# 3814.1287482832076, 8.997013262136385
# 5023.387467245986, 4.0112403797335485
# 5023.387467245986, 3.7477489555614536
# 5506.349675300583, 2.7709133868033526
# 5764.973856945102, 2.750073253755905
# 6616.038238730782, 2.209338675153079
# 7087.585955207867, 1.829342743352861
# 7087.585955207867, 1.709176391024469
# 7420.478202447991, 2.455635676842897
# 7087.585955207867, 2.60851562580635
# 7420.478202447991, 4.358572378647118
# 7949.361111716614, 3.6362667744732
# 7592.7424932925815, 6.12188611096405
# 8713.634427317096, 2.750073253755905
# 8915.918972106352, 2.94342150808186
# 10712.734267037013, 3.921413349466943
# 12579.627292639776, 5.7197500290311005
# 14436.71684596138, 3.9810717055349722
# 14109.176330555945, 2.60851562580635
# 14436.71684596138, 2.511886431509581
# 14436.71684596138, 2.209338675153079
# 11476.267547614952, 2.048686048133601
# 9773.119803553482, 1.9727950393776206
# 8515.939318253828, 2.0028081235439275
# 8515.939318253828, 1.4696472784789474
# 10232.147155674891, 1.415206035721982
# 14109.176330555945, 1.761577032603788
# 16567.962503341336, 1.761577032603788
# 19013.837040719754, 1.8432055545011783
# 14109.176330555945, 1.415206035721982
# 10961.427345996415, 1.1369404155908955
# 9334.685011995358, 0.9629562524599026
# 11215.893764047418, 0.8795535687596046
# 11476.267547614952, 0.8795535687596046
# 13170.471682444066, 0.9203106043127407
# 13789.067060798448, 0.9629562524599026
# 14217.522370573159, 0.9133889172148337
# 19455.23785946669, 0.8533899832601621
# 21325.71762956234, 0.9343117342198284
# 13476.220436441672, 0.44921993057509424
# """
# with open('lum_vs_nu.dat', 'w') as file:
#     file.write(data.strip())

import numpy as np
import matplotlib.pyplot as plt
from data import read_hdf5_data
# from regression import read_data, calculate_lum_from_teff_logg
from curvefit_params import power_spectrum_model
from scipy.optimize import curve_fit
# import read_tess_data
# from read_tess_data import light_curve_to_power_spectrum
import h5py
# from astropy.io import fits
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

def read_light_curve(file_path):
	"""
	takes a fits file and reads the relevant information  
	returns light curve output: time, photon count 
	"""
	with fits.open(file_path) as hdul:
		try:
			hdul.info()
			data = hdul[1].data
			time = data['TIME']
			flux = data['PDCSAP_FLUX']
			# mask = ~np.isnan(time) & ~np.isnan(flux)
			mask = ~np.isnan(flux)
			time_clean = time[mask]
			flux_clean = flux[mask]
			return np.array(time_clean), np.array(flux_clean)
		except:
			print('File corrupted: ', file_path)
			return None
          
def convert_epersec_to_mag(flux):
    """
    converts flux from e-/s to magnitude
    """
    return [-2.5*np.log10(i) + 20.44 for i in flux]


def save_power_freq_info(h5_file_path):
    tic_id, teff, logg, Msp = read_data()
    db = read_tess_data.Database('/mnt/home/neisner/ceph/latte/output_LATTE/tess_database.db')
    with h5py.File(h5_file_path, 'w') as h5f:
        for tic_id, t, g, m in zip(tic_id, teff, logg, Msp):
            sectorids, lcpaths, tppaths = db.search(tic_id)
            if lcpaths != 0:
                obs_id = 0
                for filepath in lcpaths:
                    print('looking for filepath: ', filepath)
                    if read_light_curve(filepath) is not None:
                        time, flux = read_light_curve(filepath)
                        flux = convert_epersec_to_mag(flux)
                        # flux = [i/np.median(flux) for i in flux]
                        freq, power = light_curve_to_power_spectrum(time, flux)
                        dataset_name = f'TIC_{tic_id}_{obs_id}_Power'
                        h5f.create_dataset(dataset_name, data=power)
                        if 'Frequency' not in h5f:
                            h5f.create_dataset('Frequency', data=freq)
                        h5f[dataset_name].attrs['TIC_ID'] = tic_id
                        h5f[dataset_name].attrs['Teff'] = t
                        h5f[dataset_name].attrs['logg'] = g
                        h5f[dataset_name].attrs['Msp'] = m
                        obs_id += 1

h5_file_path = '/mnt/sdceph/users/rzhang/tessOregression_magunits.h5'
# save_power_freq_info(h5_file_path)

def make_linear_regression(filename):
    data = np.loadtxt(filename, delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    x_log = np.array([np.log10(i) for i in x])
    y_log = np.array([np.log10(i) for i in y])

    A = np.vstack([x_log, np.ones(len(x_log))]).T
    m, c = np.linalg.lstsq(A, y_log, rcond=None)[0]
    print('Exponent (m):', m)
    print('Coefficient (a) in log scale:', c)

    x_fit = np.linspace(min(x), max(x), 400)
    # y_fit = 10**c * x_fit**m  
    y_fit = 10**(m*np.log10(x_fit) + c)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue')
    plt.plot(x_fit, y_fit, 'r')
    plt.title('Linear Regression Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Luminosity (L/L$\odot$)')
    plt.ylabel(r'$\alpha_0$ ($\mu$mag)')
    plt.grid(True)
    plt.savefig('linear_regression_plot.png')
    plt.close()
    return c, m

def calculate_lum_from_y(y, c, m):
    x_log = (np.log10(y)-c)/m
    return x_log

def preprocess_data(power, Teff, logg, Msp, freq):
    Teff = [float(t) for t in Teff]
    logg = [float(l) for l in logg]
    Msp = [float(m) for m in Msp]

    power_tensor = torch.tensor(np.array(power.tolist(), dtype=np.float32))
    
    # normalized labels tensor for Teff, logg, and Msp
    labels_tensor = torch.tensor(list(zip(Teff, logg, Msp)), dtype=torch.float32)
    
    return power_tensor, labels_tensor

def create_dataloaders(power_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(power_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

def calculate_lum_from_teff_logg(Teff, logg, Msp):
    G = 6.67e-8
    sigma = 5.67e-5
    g_in_solar_mass = 1.989e33
    erg_in_solar_lum = 3.826e33
    g = 10 ** logg
    Msp_cgs = Msp * g_in_solar_mass
    Teff_K = Teff * 1000
    L = 4 * np.pi * G * Msp_cgs * sigma * Teff_K**4 / g
    L_solar = L / erg_in_solar_lum
    logL_solar = np.log10(L_solar)
    return logL_solar

def extract_data_from_dataset(dataset):
    params_list = []
    labels_list = []
    teff = []
    logL = []
    alpha0 = []
    nuchar = []
    gamma = []
    Cw = []
    for feature, label in test_dataset:
        featurelist = feature.tolist()
        labels_list.append(label.tolist())
        teff.append(label[0])
        logL.append(calculate_lum_from_teff_logg(float(label[0]), float(label[1]), float(label[2])))
        params, params_covariance = curve_fit(power_spectrum_model, frequencies, featurelist, p0=[0.0003, 5, 3, 2e-5], bounds=([0, 0, 0, 0], [1e8, 1e8, 1e8, 1e8]))
        params_list.append(params)
        alpha0.append(params[0])
        nuchar.append(params[1])
        gamma.append(params[2])
        Cw.append(params[3])
    return np.array(params_list), np.array(labels_list), np.array(teff), np.array(logL), np.array(alpha0), np.array(nuchar), np.array(gamma), np.array(Cw)

power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression_magunits.h5')  
power_tensor, labels_tensor = preprocess_data(power, Teff, logg, Msp, frequencies)
batch_size = 32
train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size)

train_params, train_labels, train_teff, train_logL, train_alpha0, train_nuchar, train_gamma, train_Cw = extract_data_from_dataset(train_dataset)
test_params, test_labels, test_teff, test_logL, test_alpha0, test_nuchar, test_gamma, test_Cw = extract_data_from_dataset(test_dataset)

# apply linear regression
# Teff and nuchar
reg_teff_nuchar = LinearRegression()
reg_teff_nuchar.fit(train_nuchar.reshape(-1, 1), train_teff)
teff_pred = reg_teff_nuchar.predict(test_nuchar.reshape(-1, 1))
print("Teff vs. nuchar R^2 Score:", r2_score(test_teff, teff_pred))
print("Teff vs. nuchar MSE:", mean_squared_error(test_teff, teff_pred))
print('test teff', test_teff)
print('teff pred', teff_pred)

# logL and alpha0
reg_logL_alpha0 = LinearRegression()
reg_logL_alpha0.fit(train_alpha0.reshape(-1, 1), train_logL)
logL_pred = reg_logL_alpha0.predict(test_alpha0.reshape(-1, 1))
print("logL vs. alpha0 R^2 Score:", r2_score(test_logL, logL_pred))
print("logL vs. alpha0 MSE:", mean_squared_error(test_logL, logL_pred))
print('test logL', test_logL)
print('logL pred', logL_pred)

# apply lasso and ridge regression to training dataset
ridge = Ridge(alpha=1.0)
ridge.fit(train_params, train_labels)
y_pred_ridge = ridge.predict(test_params)
print("Ridge Regression:", r2_score(test_labels, y_pred_ridge))
# print('test labels', test_labels)
# print('y pred ridge', y_pred_ridge)
teff_pred_ridge = []
logL_pred_ridge = []
for pred in y_pred_ridge:
    teff_pred_ridge.append(pred[0])
    logL_pred_ridge.append(calculate_lum_from_teff_logg(pred[0], pred[1], pred[2]))
print('teff pred ridge', teff_pred_ridge)
print('logL pred ridge', logL_pred_ridge)

lasso = Lasso(alpha=0.1)
lasso.fit(train_params, train_labels)
y_pred_lasso = lasso.predict(test_params)
print("Lasso Regression:", r2_score(test_labels, y_pred_lasso))
# print('test labels', test_labels)
# print('y pred lasso', y_pred_lasso)


# # using linear correlation from Anders et al. 2023 as reference to make predictions on test dataset
# power_list = []
# actual_lum_list = []
# for feature, label in test_dataset:
#     power_list.append(feature.tolist())
#     label_list = label.tolist()
#     actual_lum_list.append(calculate_lum_from_teff_logg(label_list[0], label_list[1], label_list[2]))
# # print('power_list', power_list[0])
# print('actual_lum_list', actual_lum_list)
# # print('frequencies', frequencies)

# pred_lum_list = []
# c, m = make_linear_regression('lum_vs_nu.dat')
# i = 0
# for p in power_list:
#     if i != 9: # 10th one is an outlier curve fit
#         params, params_covariance = curve_fit(power_spectrum_model, frequencies, p, p0=[0.0003, 5, 3, 2e-5], bounds=([0, 0, 0, 0], [1e8, 1e8, 1e8, 1e8]))
#         print('params', params)
#         alpha0, nu_char, gamma, Cw = params
#         print("Fitted parameters:")
#         print(f"alpha0 (Amplitude): {alpha0}")
#         print(f"nu_char (Characteristic Frequency): {nu_char}")
#         print(f"gamma (Shape factor): {gamma}")
#         print(f"Cw (Constant offset): {Cw}")
#         # pred_lum = calculate_lum_from_y(alpha0*10**6, c, m)
#         pred_lum = calculate_lum_from_y(nu_char, c, m)
#         pred_lum_list.append(pred_lum)
#     elif i == 9:
#         plt.scatter(frequencies, p)
#         plt.plot(frequencies, power_spectrum_model(frequencies, *params), label='Fitted function', color='red')
#         plt.xlabel('Frequency')
#         plt.ylabel('Amplitude Power')
#         plt.yscale('log')
#         plt.xscale('log')
#         plt.savefig('power_mag_units.png')
#         plt.close()
#     i += 1

# actual_lum_list.pop(9)
# print('pred lum list', pred_lum_list)
# print('MSE', mean_squared_error(actual_lum_list, pred_lum_list))

# plt.figure(figsize=(10, 6))
# plt.scatter(actual_lum_list, pred_lum_list, alpha=0.3)
# plt.xlabel('Actual logL')
# plt.ylabel('Predicted logL')
# plt.title('Benchmark Predicted vs Actual Spectroscopic logL')
# plt.plot([min(actual_lum_list), max(actual_lum_list)], [min(actual_lum_list), max(actual_lum_list)], 'r')
# plt.grid(True)
# plt.savefig("benchmark_pred_vs_act_logL.png")
# plt.close()