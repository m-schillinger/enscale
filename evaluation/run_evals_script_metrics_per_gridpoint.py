import argparse
from eval_metrics_funs import *
import h5py
import properscoring as ps
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from scipy.stats import rankdata
import pandas as pd
from utils import *


def eval_crps_perloc(true_data, gen_data, var = "tas", methods = None, pool_method=None, kernel_size=1):
    crps = {}
    for method in methods:
        true = torch.tensor(true_data[var].copy())
        gen = torch.tensor(gen_data[method][var].copy())
        
        crps[method] = ps.crps_ensemble(true, gen).mean(axis=0)
    return crps

# eval_rh_per_loc is in eval_metrics_funs.py

def eval_quantile_loss_perloc(true_data, gen_data, var, methods, pool_method = None, pool_quantile=0.9, quantiles = [0.1, 0.3, 0.5, 0.6, 0.9]):
    res_quantiles = {}
    for q in quantiles:
        if pool_method != "quantile":
            res_quantiles[f"QL-{pool_method}_q-{q}"] = {}
        else:
            res_quantiles[f"QL-{pool_method}-qpool{pool_quantile}_q-{q}"] = {}
    # Loop over the methods
    for method in methods:
        # Append the results to the dataframe
        if pool_method is None:
            gen = gen_data[method][var]
            true = true_data[var]
       
        for q in quantiles:
            if pool_method != "quantile":
                res_quantiles[f"QL-{pool_method}_q-{q}"][method] = quantile_loss(q, np.quantile(gen, q, axis=-1), true).mean(axis=0)
            
    return res_quantiles


def eval_pointwise_corr(true_data, gen_data, var1 = "tas", var2 = "pr", 
                        methods = None, pool_method=None, kernel_size=1, j=0):
    corrs = {}
    if methods is None:
        methods = gen_data.keys()
    
    true1 = torch.tensor(true_data[var1][:, :, :].copy())
    true2 = torch.tensor(true_data[var2][:, :, :].copy())
    
    if pool_method == "average":
        pool = torch.nn.AvgPool2d(kernel_size)
    elif pool_method == "max":
        pool = torch.nn.MaxPool2d(kernel_size)
    
    if pool_method is not None:
        true1 = pool(true1.unsqueeze(1)).squeeze(1)
        true2 = pool(true2.unsqueeze(1)).squeeze(1)
            
    correlation_array_true = np.empty_like(true1[0, :, :])
    for lat_idx in range(true1.shape[1]):
        for lon_idx in range(true1.shape[2]):
            point1 = true1[:, lat_idx, lon_idx]
            point2 = true2[:, lat_idx, lon_idx]
            correlation_coefficient = np.corrcoef(point1, point2)[0, 1]
            correlation_array_true[lat_idx, lon_idx] = correlation_coefficient
    
    for method in methods:
        gen1 = torch.tensor(gen_data[method][var1][:, :, :, j].copy())
        gen2 = torch.tensor(gen_data[method][var2][:, :, :, j].copy())
        
        if pool_method is not None:
            gen1 = pool(gen1.unsqueeze(1)).squeeze(1)
            gen2 = pool(gen2.unsqueeze(1)).squeeze(1)
        
        correlation_array_gen = np.empty_like(true1[0, :, :])
        for lat_idx in range(true1.shape[1]):
            for lon_idx in range(true1.shape[2]):
                point1 = gen1[:, lat_idx, lon_idx]
                point2 = gen2[:, lat_idx, lon_idx]
                correlation_coefficient = np.corrcoef(point1, point2)[0, 1]
                correlation_array_gen[lat_idx, lon_idx] = correlation_coefficient

        corrs[method] = correlation_array_gen
        
    # create a dataframe
    return corrs, correlation_array_true


def eval_rh_var_diffs_perloc(true_data, gen_data, var1 = "tas", var2 = "pr", methods = None,
                             primitive_standardization=False, num_samples=9, num_locs = 100):
    
    mcb_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}
    means_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}
    var_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}

    possible_indices = [(i, j) for i in range(true_data[var1].shape[-2]) for j in range(true_data[var1].shape[-1])]
    if num_locs < len(possible_indices):
        selected_indices = np.random.choice(len(possible_indices), num_locs, replace=False)
        selected_indices = [possible_indices[idx] for idx in selected_indices]
    else:
        selected_indices = possible_indices
    for i in range(true_data[var1].shape[-2]):
        for j in range(true_data[var2].shape[-1]):
            if primitive_standardization:
                ground_truth_loc_var1 = true_data[var1][..., i, j]
                ground_truth_loc_var2 = true_data[var2][..., i, j]
                mean_true_loc_var1 = ground_truth_loc_var1.mean()
                std_true_loc_var1 = ground_truth_loc_var1.std()
                mean_true_loc_var2 = ground_truth_loc_var2.mean()
                std_true_loc_var2 = ground_truth_loc_var2.std()
                
                ground_truth_loc_var1 = (ground_truth_loc_var1 - mean_true_loc_var1) / std_true_loc_var1
                ground_truth_loc_var2 = (ground_truth_loc_var2 - mean_true_loc_var2) / std_true_loc_var2
                ground_truth_loc = ground_truth_loc_var1 - ground_truth_loc_var2
                
                forecasts_loc_var1 = {method: gen_data[method][var1][..., i, j, :num_samples] for method in methods}
                forecasts_loc_var2 = {method: gen_data[method][var2][..., i, j, :num_samples] for method in methods}
                mean_gen_loc_var1 = {method: forecasts_loc_var1[method].mean() for method in methods}
                std_gen_loc_var1 = {method: forecasts_loc_var1[method].std() for method in methods}
                mean_gen_loc_var2 = {method: forecasts_loc_var2[method].mean() for method in methods}
                std_gen_loc_var2 = {method: forecasts_loc_var2[method].std() for method in methods}

                forecasts_loc_var1 = {method: (forecasts_loc_var1[method] - mean_gen_loc_var1[method]) / std_gen_loc_var1[method] for method in methods}
                forecasts_loc_var2 = {method: (forecasts_loc_var2[method] - mean_gen_loc_var2[method]) / std_gen_loc_var2[method] for method in methods}
                forecasts_loc = {method: forecasts_loc_var1[method] - forecasts_loc_var2[method] for method in methods}
            else:
                if (i, j) not in selected_indices:
                    for method in methods:
                        mcb_locs[method][i, j] = 0
                        var_locs[method][i, j] = 0
                        means_locs[method][i, j] = 0
                    continue
                ground_truth_loc = ecdf_full(true_data[var1][..., i, j]) - ecdf_full(true_data[var2][..., i, j])
                forecasts_loc = {method: ecdf_full(gen_data[method][var1][..., i, j, :num_samples], flatten=True) - 
                                    ecdf_full(gen_data[method][var2][..., i, j, :num_samples], flatten=True) 
                                    for method in methods}
            
            rh_diff_per_loc, _ = eval_rank_histogram_agg(ground_truth_loc, forecasts_loc, methods = methods, full=True)
            for method in methods:
                mcb_locs[method][i, j] = rh_diff_per_loc.loc[method, "mcb"]
                means_locs[method][i, j] = rh_diff_per_loc.loc[method, "rh-mean"]
                var_locs[method][i, j] = rh_diff_per_loc.loc[method, "rh-variance"]
                
    return mcb_locs, means_locs, var_locs
                            
def eval_quantiles_perloc(true_data, gen_data, var, methods, quantiles = [0.95], months=None, time_vals_months=None):
    res_quantiles = {}
    for q in quantiles:
        res_quantiles[f"q-{q}"] = {}
    # Loop over the methods
    for method in methods:
        # Append the results to the dataframe
        gen = gen_data[method][var]
        true = true_data[var]
        
        if months is not None:            
            gen = gen[np.isin(time_vals_months, months)]
            true = true[np.isin(time_vals_months, months)]
        for q in quantiles:
            res_quantiles[f"q-{q}"][method] = np.quantile(gen, q, axis=(0, -1))
    
    q_true = {}
    for q in quantiles:
        q_true[q] = np.quantile(true, q, axis=0)
    return res_quantiles, q_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation metrics per gridpoint.")
    parser.add_argument("--save_name", type=str, default="maybritt_full-diff-gan-wlabel", help="Output folder name")
    parser.add_argument("--run_ids", nargs="+", default=["maybritt_3step_dense-conv_v3", "maybritt_3step_dense-conv_temporal_v2"], help="List of run IDs")
    parser.add_argument("--mode", type=str, default="test_interpolation", help="Mode for loading data")
    parser.add_argument("--run_univariate", action="store_true", help="Whether to run univariate metrics")
    parser.add_argument("--run_multivariate", action="store_true", help="Whether to run multivariate metrics")
    parser.add_argument("--run_temporal", action="store_true", help="Whether to run temporal metrics")
    parser.add_argument("--run_quantiles", action="store_true", help="Whether to run quantile metrics")
    parser.add_argument("--run_index", type=int, default=0, help="Index of the GCM-RCM pair to run")
    parser.add_argument("--run_benchmarks", action="store_true", help="Whether to run benchmark metrics")
    parser.add_argument("--subfolder", type=str, default="maybritt", help="Subfolder for saving results")
    args = parser.parse_args()

    run_ids = args.run_ids
    mode = args.mode
    
    if args.mode == "test_interpolation":
        subfolder_period = 'interpolation'
    elif args.mode == "test_extrapolation":
        subfolder_period = 'extrapolation'
    else:
        raise ValueError("Mode not recognized")
    
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations("/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear")
    tuples = list(zip(gcm_list, rcm_list))
    for i, (gcm, rcm) in enumerate(tuples):
        if i != args.run_index:
            continue
        print(i, gcm, rcm)  
        
        variables = ["tas", "pr", "sfcWind", "rsds"]
        
        out_path = os.path.join('output_evals_per_loc', args.subfolder, subfolder_period, args.save_name)
        os.makedirs(f"{out_path}/crps_results", exist_ok=True)
        os.makedirs(f"{out_path}/rh_results", exist_ok=True)
        os.makedirs(f"{out_path}/ql_results", exist_ok=True)
        os.makedirs(f"{out_path}/correlation_results", exist_ok=True)
        os.makedirs(f"{out_path}/acf_results", exist_ok=True)
        os.makedirs(f"{out_path}/quantile_results", exist_ok=True)
        
        gcm = gcm_list[i]
        rcm = rcm_list[i]
        if args.run_benchmarks:
            true_data, gen_data = load_data(gcm, rcm, variables, mode=args.mode, run_ids=args.run_ids,
                                        load_diffusion=True, load_benchmarks=True, load_gan=True)
        else:
            true_data, gen_data = load_data(gcm, rcm, variables, mode=args.mode, run_ids=args.run_ids,
                                        load_diffusion=False, load_benchmarks=False, load_gan=False)
        
        #gen_data["maybritt_marg"] = gen_data.pop("maybritt_3step_dense-conv_v3")
        #gen_data["maybritt_temp"] = gen_data.pop("maybritt_3step_dense-conv_temporal_v2")
        #methods = ["maybritt_marg", "maybritt_temp", "corrdiff", "gan"]
        methods = gen_data.keys()
        print(methods)

        if args.run_univariate:
            for j, var in enumerate(variables):
                crps_results = eval_crps_perloc(true_data, gen_data, var=var, methods=methods, pool_method=None)
                for method in methods:
                    np.save(f"{out_path}/crps_results/{gcm}_{rcm}_{var}_{method}_crps.npy", crps_results[method])
                    
                mcb_locs, means_locs, var_locs = eval_rh_perloc(true_data, gen_data, var = var, methods = methods, num_locs=128*128)#128*128)
                for method in methods:                            
                    np.save(f"{out_path}/rh_results/{gcm}_{rcm}_{var}_{method}_mcb.npy", mcb_locs[method])
                    np.save(f"{out_path}/rh_results/{gcm}_{rcm}_{var}_{method}_means.npy", means_locs[method])
                    np.save(f"{out_path}/rh_results/{gcm}_{rcm}_{var}_{method}_var.npy", var_locs[method])

                #ql_results = eval_quantile_loss_perloc(true_data, gen_data, var = var, methods = methods, pool_method=None, quantiles=[0.9])["QL-None_q-0.9"]
                #for method in methods:
                #    np.save(f"output_evals_per_loc/ql_results/{gcm}_{rcm}_{var}_{method}_ql0.9.npy", ql_results[method])

        if args.run_quantiles:
            time_vals = get_dates(rcm, gcm, mode="test_interpolation")
            time_vals_months = pd.to_datetime(time_vals).month

            for j, var in enumerate(variables):             
                quantile_results, quantile_true = eval_quantiles_perloc(true_data, gen_data, var=var, methods=methods, 
                                                                        quantiles=[0.01, 0.05, 0.95, 0.99], months = [6,7,8], time_vals_months=time_vals_months)
                for q in [0.01, 0.05, 0.95, 0.99]:
                    for method in methods:
                        np.save(f"{out_path}/quantile_results/{gcm}_{rcm}_{var}_{method}_summer_q{q}.npy", quantile_results[f"q-{q}"][method])
                    np.save(f"{out_path}/quantile_results/{gcm}_{rcm}_{var}_true_summer_q{q}.npy", quantile_true[q])

                quantile_results, quantile_true = eval_quantiles_perloc(true_data, gen_data, var=var, methods=methods,
                                                                        quantiles=[0.01, 0.05, 0.95, 0.99], months=[12, 1, 2], time_vals_months=time_vals_months)
                for q in [0.01, 0.05, 0.95, 0.99]:
                    for method in methods:
                        np.save(f"{out_path}/quantile_results/{gcm}_{rcm}_{var}_{method}_winter_q{q}.npy", quantile_results[f"q-{q}"][method])
                    np.save(f"{out_path}/quantile_results/{gcm}_{rcm}_{var}_true_winter_q{q}.npy", quantile_true[q])


        if args.run_multivariate:
            variable_pairs = [('tas', 'pr'), ('pr', 'sfcWind'), ('sfcWind', 'rsds')] #, ('rsds', 'tas')]
            for j, (var1, var2) in enumerate(variable_pairs):
                corrs, correlation_array_true = eval_pointwise_corr(true_data, gen_data, var1 = var1, var2 = var2, methods = methods, pool_method=None, kernel_size=1, j=0)
                
                for method in methods:
                    np.save(f"{out_path}/correlation_results/{gcm}_{rcm}_{var1}_{var2}_{method}_corr.npy", corrs[method])
                np.save(f"{out_path}/correlation_results/{gcm}_{rcm}_{var1}_{var2}_true_corr.npy", correlation_array_true)

                """
                mcb_locs, means_locs, var_locs = eval_rh_var_diffs_perloc(true_data, gen_data, var1=var1, var2=var2,
                                        methods=methods, num_locs=128*28)
                
                for method in methods:
                    np.save(f"output_evals_per_loc/rh_results/{gcm}_{rcm}_{var1}_{var2}_{method}_mcb.npy", mcb_locs[method])
                    np.save(f"output_evals_per_loc/rh_results/{gcm}_{rcm}_{var1}_{var2}_{method}_means.npy", means_locs[method])
                    np.save(f"output_evals_per_loc/rh_results/{gcm}_{rcm}_{var1}_{var2}_{method}_var.npy", var_locs[method])
                """
        if args.run_temporal:
            for j, var in enumerate(variables):
                acf_lag1_true, acf_gens, acf_errors = eval_acf_spatial(true_data, gen_data, var=var, methods=methods, pool_method=None, kernel_size=1)

                for method in methods:
                    np.save(f"{out_path}/acf_results/{gcm}_{rcm}_{var}_{method}_acf_lag1.npy", acf_gens[method])

                np.save(f"{out_path}/acf_results/{gcm}_{rcm}_{var}_true_acf_lag1.npy", acf_lag1_true)
