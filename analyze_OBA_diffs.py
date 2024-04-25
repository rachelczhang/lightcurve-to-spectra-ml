import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

def load_data(filename):
	"""
	Directly read file saved from read_tess_data.py
	"""
	data = pd.read_hdf(filename, 'df')
	O_stars = pd.DataFrame(data['O'].iloc[0]).T
	B_stars = pd.DataFrame(data['B'].iloc[0]).T
	A_stars = pd.DataFrame(data['A'].iloc[0]).T
	freq = data['freq'][0]
	return O_stars, B_stars, A_stars, freq

def get_average_periodogram(power_df):
	"""
	takes the dataframe of powers of a particular spectral type and returns the average power and Q1 and Q3 percentiles 
	"""
	avg_power = power_df.mean(axis=1)
	q1 = power_df.quantile(0.1, axis=1)
	q3 = power_df.quantile(0.9, axis=1)
	return avg_power, q1, q3

def get_periodogram_variance(power):
	"""
	takes the list of powers of a particular spectral type and returns the variance of the power
	"""
	df = pd.DataFrame(power)
	power_variance = df.var()
	return power_variance

def plot_categories_periodograms(freq, power_list, iqrs):
	"""
	plots the periodograms of the different spectral type categories
	"""
	count = 0
	for power_spectrum in power_list:
		if count == 0:
			plt.plot(freq, power_spectrum, linewidth=0.5, label='O Star', color='blue')
			plt.fill_between(freq, iqrs[0], iqrs[1], color='blue', alpha=0.2)#, label='O Star')
		elif count == 1:
			plt.plot(freq, power_spectrum, linewidth=0.5, label='B Star', color='red')
			plt.fill_between(freq, iqrs[2], iqrs[3], color='red', alpha=0.2)#, label='B Star')
		elif count == 2:
			plt.plot(freq, power_spectrum, linewidth=0.5, label='A Star', color='orange')
			plt.fill_between(freq, iqrs[4], iqrs[5], color='orange', alpha=0.2)#, label='A Star')
		count += 1
	plt.xlabel('Frequency [1/d]')
	plt.ylabel('Amplitude Power')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	# if avg:
	plt.savefig('avg_OBA_periodograms.png')
	# else:
	# 	plt.savefig('var_OBA_periodograms.png')
	plt.clf()

O_stars, B_stars, A_stars, freq = load_data('OBApowers.h5')
avg_O_power, O_q1, O_q3 = get_average_periodogram(O_stars)
avg_B_power, B_q1, B_q3 = get_average_periodogram(B_stars)
avg_A_power, A_q1, A_q3 = get_average_periodogram(A_stars)
plot_categories_periodograms(freq, [avg_O_power, avg_B_power, avg_A_power], [O_q1, O_q3, B_q1, B_q3, A_q1, A_q3])
# O_power_var = get_periodogram_variance(O_stars)
# B_power_var = get_periodogram_variance(B_stars)
# A_power_var = get_periodogram_variance(A_stars)
# plot_categories_periodograms(freq, [O_power_var, B_power_var, A_power_var], avg=False)
