import os
import re
import numpy as np
import pandas as pd
import h5py
import torch
import xarray as xr
from multiprocessing import Pool
import multiprocessing
from isodisreg import idr
import pdb
from utils import get_rcm_gcm_combinations

def idr_gridpoint(true, gen, loc_i, loc_j, gen_test=None, subsample_size=1):
    ml_pred = gen[:, loc_i, loc_j].flatten()
    X = pd.DataFrame({"X": ml_pred})
    y = true[:, loc_i, loc_j].flatten()
    # y_df = pd.DataFrame({"Y": true[var][:, loc_i, loc_j].flatten()})

    if subsample_size < 1:
        idx = np.random.choice(len(y), size=int(subsample_size * len(y)), replace=False)
        X = X.iloc[idx]
        y = y[idx]
    # compute idr
    fit = idr(y = y, X = X)

    # fit idr / make prediction
    if gen_test is not None:
        ml_pred_test = gen_test[:, loc_i, loc_j].flatten()
        X_test = pd.DataFrame({"X": ml_pred_test})
        preds = fit.predict(X_test)
    else:
        preds = fit.predict(X)
    return preds

def get_ensemble_preds_map(preds, U_mat=None):
    # at each point in time, interpolate from ECDF in this location and time point to the desired probabilities
    # U a matrix of shape (n_timesteps, n_samples)
    
    def pred_timestep(args):            
        single_pred, t = args
        U_mat_loc_t = U_mat[t, :] 
        return np.interp(U_mat_loc_t, single_pred.ecdf, single_pred.points)
        
    args = [(preds.predictions[t], t) for t in range(len(preds.predictions))]    
    return np.vstack(list(map(pred_timestep, args))).squeeze()


# Function to handle each (rcm, gcm, var) combination
def process_variable(rcm, gcm, var, data_dir, dir_det, store_path, q_type, n_samples, det_model):
    print(f"Processing {rcm}, {gcm}, {var}")

    # Load generated data (deterministic model output)
    idx = get_run_index(rcm, gcm)
    if det_model == "linear_ridge":
        with h5py.File(f"{dir_det}/{var}_ridge_{rcm}_{gcm}_train-period.h5", 'r') as f:
            gen_data = np.flip(f[var][:], 1)
    else:
        gen_data = torch.load(f"{dir_det}/{var}_idx{idx}_train.pt").view(-1, 128, 128).numpy()

    # Load true data
    true_data = np.flip(xr.open_dataset(os.path.join(data_dir, "train", f"{var}_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc"))[var].data, 1)

    # Load test data

    if det_model == "linear_ridge":
        with h5py.File(f"{dir_det}/{var}_ridge_{rcm}_{gcm}_int-period.h5", 'r') as f:
            gen_data_test = np.flip(f[var][:], 1)
        len_test_int = gen_data_test.shape[0]
        with h5py.File(f"{dir_det}/{var}_ridge_{rcm}_{gcm}_ext-period.h5", 'r') as f:
            gen_data_test = np.concatenate([gen_data_test, np.flip(f[var][:], 1)], axis=0)
    else:
        gen_data_test = torch.load(f"{dir_det}/{var}_idx{idx}_inter.pt").view(-1, 128, 128).numpy()
        len_test_int = gen_data_test.shape[0]
        gen_data_test = np.concatenate([gen_data_test, torch.load(f"{dir_det}/{var}_idx{idx}_extra.pt").view(-1, 128, 128).numpy()], axis=0)
    
    len_test_ext = gen_data_test.shape[0] - len_test_int
    len_test_full = len_test_int + len_test_ext

    # Run IDR for each grid point
    U_mat = np.random.uniform(0, 1, size=(len_test_full, 128, 128, 100))
    results = []
    for loc_i in range(true_data.shape[1]):
        for loc_j in range(true_data.shape[2]):
            print(f"IDR gridpoint {loc_i} and {loc_j}", end='\r')
            #if loc_i < 3 and loc_j < 3:
            preds = idr_gridpoint(true_data, gen_data, loc_i, loc_j, gen_data_test, subsample_size=0.1)
            U_mat_loc = U_mat[:, loc_i, loc_j, :]
            idr_preds = get_ensemble_preds_map(preds, U_mat=U_mat_loc)
            results.append(idr_preds)
            #else:
            #    results.append(np.zeros((len_test_full, n_samples)))
    
    # have 100 generations, get only the first
    idr_preds_full = np.array(results).reshape(true_data.shape[1], true_data.shape[2], len_test_full, 100).transpose(2, 0, 1, 3)
    idr_preds = idr_preds_full[..., :n_samples]

    # Save results separately for each test period
    save_path_int = f'{var}_idr_random_subsample_{rcm}_{gcm}_int-period.h5'
    save_path_ext = f'{var}_idr_random_subsample_{rcm}_{gcm}_ext-period.h5'
    with h5py.File(os.path.join(store_path, save_path_int), 'w') as f:
        f.create_dataset(var, data=idr_preds[:len_test_int])
    with h5py.File(os.path.join(store_path, save_path_ext), 'w') as f:
        f.create_dataset(var, data=idr_preds[len_test_int:])
        
    # get quantiles for each location
    qs = [0.1, 0.5, 0.9]
    for q in qs:
        q_samples = np.quantile(idr_preds_full, q, axis=-1)
        # torch.save(q_samples, save_dir_samples + f'idx{k}_quantile-pw{q}.pt')
        
        with h5py.File(os.path.join(store_path,  f'{var}_idr_random_subsample_{rcm}_{gcm}_int-period_quantile-pw{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_samples[:len_test_int])
        with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_ext-period_quantile-pw{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_samples[len_test_int:])
        
        q_sp_mean = np.quantile(idr_preds_full.mean(axis=(-2, -3)), q, axis = -1)
        # torch.save(q_sp_mean, save_dir_samples + f'idx{k}_quantile-sp-mean{q}.pt')
        
        with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_int-period_quantile-sp-mean{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_sp_mean[:len_test_int])
        with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_ext-period_quantile-sp-mean{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_sp_mean[len_test_int:])
        
        q_sp_max = np.quantile(idr_preds_full.max(axis=(-2, -3)), q, axis = -1)
        
        with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_int-period_quantile-sp-max{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_sp_max[:len_test_int])
        with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_ext-period_quantile-sp-max{q}.h5'), 'w') as f:
            f.create_dataset(var, data=q_sp_max[len_test_int:])
        # torch.save(q_sp_mean, save_dir_samples + f'idx{k}_quantile-sp-max{q}.pt')
        
    q_cond_mean = idr_preds_full.mean(axis=-1)
    # torch.save(q_sp_mean, save_dir_samples + f'idx{k}_cond-mean.pt')
    # torch.save(samples, save_dir_samples + f'idx{k}_quantiles.pt')

    with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_int-period_cond-mean.h5'), 'w') as f:
        f.create_dataset(var, data=q_cond_mean[:len_test_int])
    with h5py.File(os.path.join(store_path, f'{var}_idr_random_subsample_{rcm}_{gcm}_ext-period_cond-mean.h5'), 'w') as f:
        f.create_dataset(var, data=q_cond_mean[len_test_int:])
    

# Main function for parallel execution
if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    
    data_dir = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    dir_det = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/nn_det_per_variable/burnin24_v1"
    det_model = "nn_det_per_variable"
    store_path = f"/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/{det_model}_idr"
    os.makedirs(store_path, exist_ok=True)
    q_type = "random-quantiles-time-space"
    variables = ["tas", "pr", "rsds", "sfcWind"]

    # Prepare quantiles matrix
    n_samples = 10

    # Gather RCM-GCM combinations
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(data_dir)
    
    # Define a function to obtain run index for GCM-RCM combinations
    def get_run_index(rcm, gcm):
        rcm_gcms = [(rcm_list[i], gcm_list[i]) for i in range(len(rcm_list))]
        return np.where([rcm_gcms[i] == (rcm, gcm) for i in range(len(rcm_gcms))])[0][0]
    
    # Set up the arguments for parallel execution
    args = [(rcm, gcm, var, data_dir, dir_det, store_path, 
             # q_type, n_samples, det_model) for rcm, gcm in zip(rcm_list, gcm_list) for var in variables]
             q_type, n_samples, det_model) for rcm, gcm in zip(rcm_list[1:], gcm_list[1:]) for var in variables]

    # Execute sequentially
    #for arg in args:
    #    process_variable(*arg)
    # Execute in parallel
    with Pool(processes=min(int(cpu_count // 2), len(args) // 3)) as pool:
        pool.starmap(process_variable, args)
