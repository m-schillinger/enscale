import xarray as xr
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import scipy
# Specify the directory path
directory_path = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/train'
path_test_interpolation = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/interpolation/'
path_test_extrapolation = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/extrapolation/'
store_path = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/analogue_dist_matrix"

#var = "tas"
#rcm = "RegCM4-6"
#gcm = "CNRM-CM5"
#hr = xr.open_dataset(os.path.join(directory_path, "{}_day_EUR-11_{}_{}_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc".format(var, rcm, gcm)))

def load_lr(gcm, variables, directory_path, path_test_interpolation, path_test_extrapolation, sqrt_transform=False):
    # Create an empty dictionary to store the variables
    data_dict = {}

    # Loop through the variables
    for var in variables:
        # Open the lr file for the variable
        file_path = os.path.join(directory_path, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_train-period.nc".format(var, gcm))
        train_data = xr.open_dataset(file_path)
        
        # Open the files in path_test_interpolation
        file_path = os.path.join(path_test_interpolation, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_2030-2039.nc".format(var, gcm))
        interpolation_file = xr.open_dataset(file_path)
        
        # Open the files in path_test_extrapolation
        file_path = os.path.join(path_test_extrapolation, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_2090-2099.nc".format(var, gcm))
        extrapolation_file = xr.open_dataset(file_path)
        
        # Concatenate the files along the time axis
        data = xr.concat([train_data, interpolation_file, extrapolation_file], dim='time')
        
        # Add the variable data to the dictionary
        # data_dict[var] = (data[var] - data[var].mean()) / data[var].std() - irrelevant if I use ranks later
        
        if var in ["pr", "sfcWind"] and sqrt_transform:
            data_dict[var] = np.sqrt(data[var])
        else:
            data_dict[var] = data[var]            

    # Create an xarray dataset with multiple variables
    dataset_lr = xr.Dataset(data_dict)
    dataset_lr = dataset_lr.sortby('time')
    
    return dataset_lr

def compute_dist_matrices(data, variables = ['tas', 'pr', 'sfcWind', 'rsds'], time_thresh = 30):
    dist_matrices = {}
    for var in variables:
        dist_matrices[var] = np.zeros((len(data.time), len(data.time)))

    # Loop through the time points
    # Convert time to day of year
    day_of_year = data.time.dt.dayofyear.values

    for var in variables:
        print("Processing variable", var)
        for i in range(len(data.time)):
            print(f"Processing index {i}", end='\r')
            # Compute differences in day of year
            diffs = np.abs(day_of_year[i] - day_of_year[i:])
            diffs = np.minimum(diffs, 366 - diffs)
            
            # Find indices where diff <= time_thresh
            indices = np.where(diffs <= time_thresh)[0]

            # Compute distances
            pw_dists = np.tile(data[var][i], (len(indices), 1, 1)) - data[var][i:][indices]
            distances = np.linalg.norm(pw_dists, axis=(1, 2))

            # Update dist_matrices
            dist_matrices[var][i, i:][indices] = distances
            dist_matrices[var][i:, i][indices] = distances
                        
    return dist_matrices
    
if __name__ == '__main__':
    for gcm in ["CNRM-CM5", "MPI-ESM-LR", "MIROC5"]:
    # for gcm in ["MPI-ESM-LR", "MIROC5"]:
    # for gcm in ["MIROC5"]:
        print(f"Processing {gcm}")
        # List of variables
        # variables = ['tas', 'pr', 'sfcWind', 'rsds']
        variables = ["psl"]

        dataset_lr = load_lr(gcm, variables, directory_path, path_test_interpolation, path_test_extrapolation, sqrt_transform=True)

        time_thresh = 50
        dist_matrices = compute_dist_matrices(dataset_lr, variables=variables, time_thresh=time_thresh)

        # Save all distance matrices per variable
        # for var in ["pr", "tas", "rsds", "sfcWind"]:
        for var in ["psl"]:
            # Convert dist_matrices[var] to a sparse matrix
            sparse_matrix = scipy.sparse.csr_matrix(dist_matrices[var])

            # Save the sparse matrix
            scipy.sparse.save_npz(os.path.join(
                store_path,
                'dist_matrix_seasonal-{}-days_sparse_{}_{}_sqrt.npz'.format(time_thresh, gcm, var)), sparse_matrix)
            