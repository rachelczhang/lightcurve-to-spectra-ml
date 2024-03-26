import glob
from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np
from astropy.timeseries import LombScargle
import astroquery
from astroquery.mast import Catalogs
from sqlite3 import Error
import sqlite3 
import os

base_directory = '/mnt/home/neisner/ceph/latte/output_LATTE/data/'
TIC = '*306491594*'
file_extension = '_lc.fits'
searchpattern = f'{base_directory}/**/{TIC}*{file_extension}'

datapath = '/mnt/home/neisner/ceph/latte/output_LATTE/'
DB_FILE = datapath + 'tess_database.db'

class Database:
	def __init__(self, db_file):
		self.connection = None
		self.db_file = db_file
		# self.re_lc = re.compile('data/s000(\d+)/tess\d{13}-s(\d{4})-(\d+)-\d+-s_lc.fits')
		self.open_connection()

	def __del__(self):
		self.close_connection()

	def open_connection(self):
		try:
			self.connection = sqlite3.connect(self.db_file)
			print("Database file opened, SQLite version:", sqlite3.version)
		except Error as e:
			print("Database connect failed with error", e)

	def close_connection(self):
		if self.connection:
			self.connection.close()

	# tid needs to be a int. Just cast it this way it takes care of leading 0's
	def search(self, tic_id):
		sql_select_by_tic = """ SELECT tic_id,sector_id,tp_filename FROM fits
							WHERE tic_id = ? """
		try:
			c = self.connection.cursor()
			c.execute(sql_select_by_tic, (tic_id,))
			row = c.fetchall() # change this to make it itterate over multiple 

			if len(row) == 0:
				print('Could not find TIC ID', tic_id, "in database")
				return [0,0,0]
			
			# sector = ([i[1] for i in row])
			# tp_path = ([i[2] for i in row])
			# lc_path = [path.replace('tp.fits', 'lc.fits') for path in tp_path]

			sector_paths = []
			lc_full_paths = []
			tp_full_paths = []

			for r in row:
				sector_id = r[1]
				tp_filename = r[2]
				sector_number = tp_filename.split('-')[1]
				tic_id_portion = tp_filename.split('-')[2][:9]
				sub_directory = f"{sector_number}/{tic_id_portion}/"
				full_directory_path = base_directory + sub_directory #gos.path.join(base_directory, sub_directory)
				lc_filename = tp_filename.replace('tp.fits', 'lc.fits')
				lc_full_path = full_directory_path + lc_filename #os.path.join(full_directory_path, lc_filename)
				tp_full_path = full_directory_path + tp_filename #os.path.join(full_directory_path, tp_filename)
				sector_paths.append(sector_id)
				lc_full_paths.append(lc_full_path)
				tp_full_paths.append(tp_full_path)

			return sector_paths, lc_full_paths, tp_full_paths
			#return (row[0][1], row[0][2], row[0][3])

		except Error as e:
			print("INSERT failed with error", e)

def query_TIC_by_spectral_type():
	queried = Catalogs.query_criteria(catalog='Tic', objType='STAR', Tmag=[8.1, 8.15], ID=306491594)
	print(queried.columns)
	filtered_query = queried[(queried['disposition'] != 'SPLIT') & (queried['Jmag']-queried['Hmag'] < 0.045) & (queried['Jmag'] - queried['Kmag'] < 0.06) & (queried['wdflag'] != 1)]
	print('ID', filtered_query['ID'])
	print('Tmag', filtered_query['Tmag'])
	print('Jmag', filtered_query['Jmag'])
	print('Hmag', filtered_query['Hmag'])
	print('Kmag', filtered_query['Kmag'])
	print('disposition', filtered_query['disposition'])
	print('wdflag', filtered_query['wdflag'])
	return filtered_query['ID']

def collect_fits(searchpattern):
	"""
	takes a string of the search pattern to search for all data files with that ID 
	returns a list of strings with each fits file
	"""
	file_paths = []
	matching_files = glob.glob(searchpattern, recursive=True)
	for file_path in matching_files:
		print(f"File: {file_path}")
		file_paths.append(file_path)
	return file_paths

def read_light_curve(file_path):
	"""
	takes a fits file and reads the relevant information  
	returns light curve output: time, photon count 
	"""
	with fits.open(file_path) as hdul:
		hdul.info()
		data = hdul[1].data
		time = data['TIME']
		flux = data['PDCSAP_FLUX']
		mask = ~np.isnan(time) & ~np.isnan(flux)
		time_clean = time[mask]
		flux_clean = flux[mask]
		flux_clean_norm = flux_clean/np.median(flux_clean)
		flux_norm = flux/np.median(flux_clean)
	# return time_clean, flux_clean_norm
	return time_clean, flux_clean_norm

def stitch_light_curve(searchpattern):
	"""
	stitches the light curve together by normalizing the flux for each light curve
	and then putting them all on the same graph
	"""
	all_times = []
	all_fluxes_norm = []
	file_paths = collect_fits(searchpattern)
	for file_path in file_paths:
		time, flux_norm = read_light_curve(file_path)
		all_times.extend(time)
		all_fluxes_norm.extend(flux_norm)
	all_times = np.array(all_times)
	all_fluxes_norm = np.array(all_fluxes_norm)
	print('time', all_times)
	print('flux', all_fluxes_norm)
	return all_times, all_fluxes_norm

def plot_light_curve(time, flux):
	plt.scatter(time, flux, s = 5)
	plt.xlabel('Time -2457000 [BTJD days]')
	plt.ylabel('Normalized Flux')
	plt.savefig('testlc.png')
	plt.clf()

def light_curve_to_power_spectrum(time, flux):
	"""
	converts the stitched light curve into a power spectrum
	"""
	freq, power = LombScargle(time, flux).autopower(normalization='standard', samples_per_peak=5, nyquist_factor=3, method='scipy')
	print(len(freq), freq.min(), freq.max())
	plt.scatter(freq, power, s=5)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Frequency [1/d]')
	plt.ylabel('Power')
	plt.savefig('testpowspec.png')
	plt.clf()

	
if __name__ == '__main__':
	# # tic_ids = query_TIC_by_spectral_type()
	# # print(tic_ids)
	# tic_ids = [306491594, 216662610, 464295672, 220322383, 294114621, 234009943, 100589904, 455675248, 427395049, 467065657, 195288472, 234648113, 178489528, \
	# 30653985, 406050497, 459532732, 339567904, 41792209, 246953610, 427395058, 464839773, 442871031, 78897024, 388935529, 59215060, 151464886, 234052684, 36557200, \
	# 95513457, 146908355, 37777866, 427451176, 80466973, 427393920, 187458882]
	# print('len tic ids', len(tic_ids))
	# # load the database - this is a quick way to search for all of the needed urls
	# db = Database(DB_FILE)
	# for tic_id in tic_ids:
	# 	sectorids, lcpaths, tppaths = db.search(tic_id)
	# 	print('lcpaths', lcpaths)
	# 	for filepath in lcpaths:
	# 		filepath=lcpaths[1]
	# 		time, flux = read_light_curve(filepath)
	# 		print('time', time)
	# 		print('len time', len(time))
	# 		print('flux', flux)
	# 		print('len flux', len(flux))
	# 		plot_light_curve(time, flux)
	# 		break
	# 	break

	time, flux = stitch_light_curve(searchpattern)
	plot_light_curve(time, flux)
	light_curve_to_power_spectrum(time, flux)