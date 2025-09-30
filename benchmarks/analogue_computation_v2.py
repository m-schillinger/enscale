import os
import numpy as np
import xarray as xr
import h5py
import warnings
import argparse
import multiprocessing
import scipy.sparse

def helper_filter_timesteps(long, short):
    short_times = short.time.values.astype('datetime64[D]')
    long_times = long.time.values.astype('datetime64[D]')

    intersection = np.intersect1d(short_times, long_times)
    # Add 12 hours to each datetime
    long_times_ns = np.array([np.datetime64(dt) for dt in intersection]) + long.time[0].dt.hour.data.astype('timedelta64[h]')
    long_filtered = long.sel(time=np.isin(long.time.values, long_times_ns))
    return long_filtered

def filter_hr_timesteps(lr, hr):
    if(len(lr.time) == len(hr.time)):
        return lr, hr
    
    elif(len(lr.time) > len(hr.time)):
        warnings.warn("Had to cut off the LR time series. This will cause problems with the distance matrix!")
        return helper_filter_timesteps(lr, hr), hr
    else:
        return lr, helper_filter_timesteps(hr, lr)

def load_hr(rcm, gcm):
    # Specify the directory path

    # List of variables
    variables = ['tas', 'pr', 'sfcWind', 'rsds']

    # Create an empty dictionary to store the variables
    data_dict = {}

    # Loop through the variables
    for var in variables:        
        file_path = os.path.join(directory_path, "{}_day_EUR-11_{}_{}_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc".format(var, rcm, gcm))
        train_data = xr.open_dataset(file_path)
        
        path_lr = os.path.join(directory_path, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_train-period.nc".format(var, gcm))
        # _, train_data = filter_hr_timesteps(xr.open_dataset(path_lr), train_data)
        assert len(xr.open_dataset(path_lr).time) == len(train_data.time)
        
        # Open the files in path_test_interpolation
        # file_path = os.path.join(path_test_interpolation, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_2030-2039.nc".format(var, gcm))
        file_path = os.path.join(path_test_interpolation, "{}_day_EUR-11_{}_{}_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc".format(var, rcm, gcm))
        interpolation_file = xr.open_dataset(file_path)
                
        path_lr = os.path.join(path_test_interpolation, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_2030-2039.nc".format(var, gcm))
        # _, interpolation_file = filter_hr_timesteps(xr.open_dataset(path_lr), interpolation_file)
        assert len(xr.open_dataset(path_lr).time) == len(interpolation_file.time)
        
        # Open the files in path_test_extrapolation
        file_path = os.path.join(path_test_extrapolation, "{}_day_EUR-11_{}_{}_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc".format(var, rcm, gcm))
        extrapolation_file = xr.open_dataset(file_path)
        
        path_lr = os.path.join(path_test_extrapolation, "{}_day_{}_r1i1p1_rcp85_EUROPE_g025_2090-2099.nc".format(var, gcm))
        # _, extrapolation_file = filter_hr_timesteps(xr.open_dataset(path_lr), extrapolation_file)
        assert len(xr.open_dataset(path_lr).time) == len(extrapolation_file.time)
        
        # Concatenate the files along the time axis
        data = xr.concat([train_data, interpolation_file, extrapolation_file], dim='time')
                
        # Replace "rlat" with "x" if present
        if "rlat" in data.coords:
            data = data.rename({"rlat": "x"})
        
        # Replace "rlon" with "y" if present
        if "rlon" in data.coords:
            data = data.rename({"rlon": "y"})
        
        # Add the variable data to the dictionary
        if "height" in data[var].coords:
            data_dict[var] = data[var].drop(("height", "lat", "lon"))
        else:
            data_dict[var] = data[var].drop(("lat", "lon"))
            
    dataset_hr = xr.Dataset(data_dict)
    dataset_hr = dataset_hr.sortby('time')
    
    return dataset_hr

def load_dist_matrix(gcm, var):
    dist_matrix_sparse = scipy.sparse.load_npz(os.path.join(
                store_path,
                'analogue_dist_matrix/dist_matrix_seasonal-50-days_sparse_{}_{}_sqrt.npz'.format(gcm, var)))
    dist_matrix = dist_matrix_sparse.toarray()
    dist_matrix[np.where(dist_matrix == 0)] = np.inf
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def convert_to_ranks(dist_matrix):
    ranks = np.zeros_like(dist_matrix)
    for i in range(dist_matrix.shape[0]):
        sorted_indices = np.argsort(dist_matrix[i])
        ranks[i, sorted_indices] = np.arange(dist_matrix.shape[1])
    return ranks

def find_analogues(dist_matrix, indices, dataset, num_neighbors = 4, num_resampling = 4, variables = ["tas", "pr", "sfcWind", "rsds"], forbidden_indices=None):
    num_days = len(indices)
    start_ind = indices[0]

    resampled_data_dict = {}
    for var in variables:
        resampled_data_dict[var] = np.zeros((num_days, num_resampling, 128, 128))

    # Loop through the first 'num_days' in LR
    for i in indices:
        # Get the distances for the current day
        distances = dist_matrix[i, :]
        
        # Find the indices of the nearest neighbors
        # neighbors = np.argsort(distances)[:num_neighbors]
        
        sorted_ind = np.argsort(distances)
        if forbidden_indices is not None:
            sorted_ind = sorted_ind[~np.in1d(sorted_ind, np.concatenate((forbidden_indices, indices)))]
        else:
            sorted_ind = sorted_ind[~np.in1d(sorted_ind,indices)]
        neighbors = sorted_ind[:num_neighbors]
        
        # Loop through the number of resampling iterations
        for j in range(num_resampling):
            # Randomly choose one neighbor out of the num_neighbours
            chosen_neighbor = np.random.choice(neighbors)
            
            # non-random version
            # chosen_neighbor = neighbors[j]
            
            # Store the resampled data in the array
            for var in variables:
                resampled_data_dict[var][i - start_ind, j, :, :] = dataset[var][chosen_neighbor, :, :]
        
        # Print the current day being processed
        print("Sampled neighbors of day {}...".format(i+1), end='\r')

    return resampled_data_dict


def process_rcm(rcm, gcm, num_neighbors, num_resampling, dist_matrices, averaged_rank_matrix):
    print(rcm)
    dataset = load_hr(rcm, gcm)
    time_stamps = dataset.time.values.astype('datetime64[D]')
    indices_int = np.where((time_stamps >= np.datetime64(f'2030-01-01T00:00:00')) & (time_stamps <= np.datetime64(f'2039-12-31T23:00:00')))[0]
    indices_ext = np.where((time_stamps >= np.datetime64(f'2090-01-01T00:00:00')) & (time_stamps <= np.datetime64(f'2099-12-31T23:00:00')))[0]

    print("indices int:", len(indices_int))
    print("indices ext:", len(indices_ext))
    
    print("Take distances based on averaged rank matrix")
    os.makedirs(os.path.join(store_path, f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-50-days-multivar'), exist_ok=True)
    # os.makedirs(os.path.join(store_path, f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-multivar'), exist_ok=True)

    # get analogues based on averaged rank matrix
    resampled_data_dict_int = find_analogues(dist_matrix=averaged_rank_matrix, indices=indices_int, dataset=dataset, 
                                             num_neighbors=num_neighbors, num_resampling=num_resampling, forbidden_indices=indices_ext)

    for var in ["tas", "pr", "sfcWind", "rsds"]:
        with h5py.File(
            os.path.join(store_path, 
            f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-50-days-multivar/{var}_analogues_{rcm}_{gcm}_int-period.h5'), 'w') as f:
            # f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-multivar/{var}_analogues_{rcm}_{gcm}_int-period.h5'), 'w') as f:
            f.create_dataset(var, data=resampled_data_dict_int[var])

    resampled_data_dict_ext = find_analogues(dist_matrix=averaged_rank_matrix, indices=indices_ext, dataset=dataset, 
                                             num_neighbors=num_neighbors, num_resampling=num_resampling, forbidden_indices=indices_int)

    for var in ["tas", "pr", "sfcWind", "rsds"]:
        with h5py.File(
            os.path.join(store_path, 
            f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-50-days-multivar/{var}_analogues_{rcm}_{gcm}_ext-period.h5'), 'w') as f:
            # f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-multivar/{var}_analogues_{rcm}_{gcm}_ext-period.h5'), 'w') as f:
            f.create_dataset(var, data=resampled_data_dict_ext[var])
    
    """
    # Specify the number of days, number of resampling iterations, and number of nearest neighbors
    os.makedirs(os.path.join(store_path, f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-singlevar'), exist_ok=True)
    # os.makedirs(os.path.join(store_path, f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-singlevar'), exist_ok=True)
    for var in ["tas", "pr", "sfcWind", "rsds"]:
        print("Take distances based on variable", var)
        # get analogues only based on dist matrix for this variable
        resampled_data_dict_int = find_analogues(dist_matrix=dist_matrices[var], indices=indices_int, dataset=dataset, 
                                                 num_neighbors=num_neighbors, num_resampling=num_resampling, variables=[var], forbidden_indices=indices_ext)

        with h5py.File(
            os.path.join(store_path, 
            f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-singlevar/{var}_analogues_{rcm}_{gcm}_int-period.h5'), 'w') as f:
            # f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-singlevar/{var}_analogues_{rcm}_{gcm}_int-period.h5'), 'w') as f:
            f.create_dataset(var, data=resampled_data_dict_int[var])

        resampled_data_dict_ext = find_analogues(dist_matrix=dist_matrices[var], indices=indices_ext, dataset=dataset, 
                                                 num_neighbors=num_neighbors, num_resampling=num_resampling, variables=[var], forbidden_indices=indices_int)

        with h5py.File(
            os.path.join(store_path, 
            f'analogue_result/resample_train-test-split/{num_neighbors}-neighbours-seasonal-singlevar/{var}_analogues_{rcm}_{gcm}_ext-period.h5'), 'w') as f:
            # f'analogue_result/resample_all-data/{num_neighbors}-neighbours-seasonal-singlevar/{var}_analogues_{rcm}_{gcm}_ext-period.h5'), 'w') as f:
            f.create_dataset(var, data=resampled_data_dict_ext[var])
    """
    
def process_rcm_wrapper(args_list):
    rcm, gcm, num_neighbors, num_resampling, dist_matrices, averaged_rank_matrix = args_list
    process_rcm(rcm, gcm, num_neighbors, num_resampling, dist_matrices, averaged_rank_matrix)

if __name__ == "__main__":
    
    directory_path = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/train'
    store_path = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/"    
    path_test_interpolation = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/interpolation/'
    path_test_extrapolation = '/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/extrapolation/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcm", type=str, help="Global Climate Model")
    parser.add_argument("--rcm", type=str, help="Regional Climate Model", nargs='+')
    parser.add_argument("--num_neighbors", type=int, help="Number of nearest neighbors")
    parser.add_argument("--num_resampling", type=int, help="Number of resampling iterations", default=9)
    args = parser.parse_args()
    gcm = args.gcm

    dist_matrices = {}
    for var in ["pr", "tas", "rsds", "sfcWind", "psl"]:
        print("Loading distance matrix for variable", var)
        dist_matrices[var] = load_dist_matrix(gcm, var)

    rank_matrices = {}
    for var in ["pr", "tas", "rsds", "sfcWind", "psl"]:
        print("Converting distance matrix to ranks for variable", var)
        rank_matrices[var] = convert_to_ranks(dist_matrices[var])
        
    # Compute the average rank matrix
    vars = list(rank_matrices.keys())
    averaged_rank_matrix = np.zeros_like(rank_matrices[vars[0]])
    for var in vars:
        averaged_rank_matrix += rank_matrices[var]
    averaged_rank_matrix /= len(vars)
    
    num_neighbors = args.num_neighbors
    num_resampling = args.num_resampling    
    pool = multiprocessing.Pool(processes=10)

    args_list = [(rcm, gcm, num_neighbors, num_resampling, dist_matrices, averaged_rank_matrix) for rcm in args.rcm]
    pool.map(process_rcm_wrapper, args_list)
    pool.close()
    pool.join()
