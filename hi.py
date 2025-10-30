# import h5py

# h5_file_path = '/mnt/sdceph/users/rzhang/tessOBAstars_all.h5'

# # Open the file in read mode and collect dataset names
# with h5py.File(h5_file_path, 'r') as h5f:
#     dataset_names = [name for name in h5f if isinstance(h5f[name], h5py.Dataset)]

# print(dataset_names)

from regression import calculate_lum_from_teff_logg

print(calculate_lum_from_teff_logg(38.3, 4.07, 18.3, False))
