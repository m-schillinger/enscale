import torch
from eval_funs import cond_energy_score_batch, cond_energy_score
import pandas as pd
import properscoring as ps
import numpy as np
from scipy.stats import rankdata
import pysteps
import re
import os
from utils import get_rcm_gcm_combinations
import pdb

def eval_es(true_data, gen_data, var = "tas", methods = None, pool_method=None, kernel_size=1):
    s_total = []
    s1 = []
    s2 = []

    if methods is None:
        methods = gen_data.keys()
    
    for method in methods:
        true = torch.tensor(true_data[var].copy()).unsqueeze(1)
        if gen_data[method][var].shape[3] > 1:
            gen1 = torch.tensor(gen_data[method][var][:, :, :, 1].copy()).unsqueeze(1)
            gen2 = torch.tensor(gen_data[method][var][:, :, :, 2].copy()).unsqueeze(1)
        else:
            gen1 = torch.tensor(gen_data[method][var][:, :, :, 0].copy()).unsqueeze(1)
            gen2 = torch.tensor(gen_data[method][var][:, :, :, 0].copy()).unsqueeze(1)

        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen1 = pool(gen1)
            gen2 = pool(gen2)
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen1 = pool(gen1)
            gen2 = pool(gen2)
            
        scores = cond_energy_score_batch(true, gen1, gen2)
            
        s_total.append(scores[0].item())
        s1.append(scores[1].item())
        s2.append(2*scores[2].item())
        
    # create a dataframe
    df = pd.DataFrame({f'ES_pool-{pool_method}_kernel-{kernel_size}': s_total, f'Score1-{pool_method}_kernel-{kernel_size}': s1, f'Score2_pool-{pool_method}_kernel-{kernel_size}': s2}, index=methods)
    return df


def eval_es_boot(true_data, gen_data, var="tas", methods=None, pool_method=None, kernel_size=1):
    # same as eval_es, but no average over the days, allows for more efficient bootstrap
    results = {f'ES_pool-{pool_method}_kernel-{kernel_size}': {}, f'Score1_pool-{pool_method}_kernel-{kernel_size}': {}, f'Score2_pool-{pool_method}_kernel-{kernel_size}': {}}

    if methods is None:
        methods = gen_data.keys()
    
    for method in methods:
        true = torch.tensor(true_data[var].copy()).unsqueeze(1)
        if gen_data[method][var].shape[3] > 1:
            gen1 = torch.tensor(gen_data[method][var][:, :, :, 1].copy()).unsqueeze(1)
            gen2 = torch.tensor(gen_data[method][var][:, :, :, 2].copy()).unsqueeze(1)
        else:
            gen1 = torch.tensor(gen_data[method][var][:, :, :, 0].copy()).unsqueeze(1)
            gen2 = torch.tensor(gen_data[method][var][:, :, :, 0].copy()).unsqueeze(1)

        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen1 = pool(gen1)
            gen2 = pool(gen2)
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen1 = pool(gen1)
            gen2 = pool(gen2)
            
        es, s1, s2 = cond_energy_score(true, gen1, gen2, full=True)
        
        results[f'ES_pool-{pool_method}_kernel-{kernel_size}'][method] = es
        results[f'Score1_pool-{pool_method}_kernel-{kernel_size}'][method] = s1
        results[f'Score2_pool-{pool_method}_kernel-{kernel_size}'][method] = 2*s2        
    return results


def eval_mse(true_data, gen_data, var = "tas", methods = None, n_samples = 3):
    mse = []
    mse_loss = torch.nn.MSELoss()
    
    # iterate over each method
    for method in methods:
        if gen_data[method][var].shape[3] < n_samples:
            n_samples = gen_data[method][var].shape[3]
        gen_mean = np.mean(gen_data[method][var][:, :, :, 0:n_samples], axis=3)
        score = mse_loss(
            torch.tensor(true_data[var].copy()), 
            torch.tensor(gen_mean.copy()))
        
        # append the scores to the dataframe
        mse.append(score.item())
        
    # create a dataframe
    df = pd.DataFrame({'mse': mse}, index=methods)
    return df

def eval_mse_boot(true_data, gen_data, var = "tas", methods = None, n_samples = 3):
    # same as eval_mse, but no average over the days, allows for more efficient bootstrap
    results = {f'MSE': {}}
    mse_loss = torch.nn.MSELoss(reduction="none")
    
    # iterate over each method
    for method in methods:
        if gen_data[method][var].shape[3] < n_samples:
            n_samples = gen_data[method][var].shape[3]
        gen_mean = np.mean(gen_data[method][var][:, :, :, 0:n_samples], axis=3)
        score = mse_loss(
            torch.tensor(true_data[var].copy()), 
            torch.tensor(gen_mean.copy()))
        
        # append the scores to the results dictionary
        results[f'MSE'][method] = score
        
    return results

def eval_mse_from_prep(true_data, gen_data_mean, var = "tas", methods = None):
    # like eval mse, but from precomputed conditional mean
    mse = []
    mse_loss = torch.nn.MSELoss()
    
    # iterate over each method
    for method in methods:
        gen_mean = gen_data_mean[method][var]
        score = mse_loss(
            torch.tensor(true_data[var].copy()), 
            torch.tensor(gen_mean.copy()))
        
        # append the scores to the dataframe
        mse.append(score.item())
        
    # create a dataframe
    df = pd.DataFrame({'mse': mse}, index=methods)
    return df

def eval_mse_from_prep_boot(true_data, gen_data_mean, var = "tas", methods = None):
    # same as eval_mse_from_prep, but no average over the days, allows for more efficient bootstrap
    results = {f'MSE': {}}
    mse_loss = torch.nn.MSELoss(reduction="none")
    
    # iterate over each method
    for method in methods:
        gen_mean = gen_data_mean[method][var]
        score = mse_loss(
            torch.tensor(true_data[var].copy()), 
            torch.tensor(gen_mean.copy()))
        
        # append the scores to the dataframe
        results[f'MSE'][method] = score
        
    return results

def eval_crps(true_data, gen_data, var = "tas", methods = None, pool_method=None, kernel_size=1):
    crps = []
    for method in methods:
        true = torch.tensor(true_data[var].copy())
        gen = torch.tensor(gen_data[method][var].copy())
        
        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen.permute(0,3,1,2)).permute(0,2,3,1)
            
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen.permute(0,3,1,2)).permute(0,2,3,1)
            
        crps.append(ps.crps_ensemble(true, gen).mean())
    df = pd.DataFrame({f'crps_pool-{pool_method}_kernel-{kernel_size}': crps}, index=methods)
    return df

def eval_crps_boot(true_data, gen_data, var = "tas", methods = None, pool_method=None, kernel_size=1):
    results = {f'CRPS_pool-{pool_method}_kernel-{kernel_size}': {}}

    for method in methods:
        true = torch.tensor(true_data[var].copy())
        gen = torch.tensor(gen_data[method][var].copy())
        
        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        crps = ps.crps_ensemble(true, gen)
        results[f'CRPS_pool-{pool_method}_kernel-{kernel_size}'][method] = crps

    return results

import pdb

def compute_rank_histogram(ground_truth, forecasts, axis=0, method="min", normalise=True):
    # assume that forecasts have an extra dimension 0 for the several samples
    # other dimensions should be equal to ground truth
    combined=np.concatenate((np.expand_dims(ground_truth, axis = axis), forecasts), axis = axis)
    ranks=np.apply_along_axis(lambda x: rankdata(x,method=method),axis,combined)
    if axis == 0:
        ranks_truth = ranks[0, :]
    elif axis == 1 or axis == -1:
        ranks_truth = ranks[:, 0]
    else:
        raise ValueError("axis must be 0, 1 or -1")
    result = np.histogram(ranks_truth, bins=np.linspace(0.5, combined.shape[axis]+0.5, combined.shape[axis]+1), density=normalise)    
    # faster
    # result = [np.bincount(ranks_truth)[1:] / len(ranks_truth)]
    return result, np.mean(ranks_truth), np.var(ranks_truth)

def compute_rank_histogram_efficient(ground_truth, forecasts, axis=0, method="min", normalise=True):
    combined = np.concatenate((np.expand_dims(ground_truth, axis = axis), forecasts), axis = axis)
    num_samples_total = combined.shape[axis]

    if axis == 0:
        first_row = combined[0, :][np.newaxis, :]  # Shape (1, n), for broadcasting
        less_than = (combined < first_row).sum(axis=0)  # Count elements smaller than first element
        equal_to = (combined == first_row).sum(axis=0)  # Count elements equal to first element
    elif axis == 1 or axis == -1:
        first_col = combined[:, 0][:, np.newaxis]  # Shape (n, 1), for broadcasting
        less_than = (combined < first_col).sum(axis=1)  # Count elements smaller than first element
        equal_to = (combined == first_col).sum(axis=1)  # Count elements equal to first element

    if method == "average":  # Mimic rankdata(method="average")
        ranks = less_than + (equal_to - 1) / 2 + 1
    elif method == "min":  # Mimic rankdata(method="min")
        ranks = less_than + 1
    elif method == "max":  # Mimic rankdata(method="max")
        ranks = less_than + (equal_to - 1)
    elif method == "dense":  # Mimic rankdata(method="dense")
        ranks =  np.unique(combined, axis=1).shape[1] - (combined > first_col).sum(axis=1)
    elif method == "randomized":
        # get a random rank between less_than + 1 and less_than + equal_to
        ranks = less_than + 1 + np.random.randint(0, equal_to, size=less_than.shape)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    freqs = np.bincount(ranks)
    if len(freqs) < num_samples_total + 1:
        freqs = np.concatenate((freqs, np.zeros(num_samples_total + 1 - len(freqs))))
    if normalise:
        return [freqs[1:] / len(ranks)], np.mean(ranks), np.var(ranks)
    else:
        return [freqs[1:]], np.mean(ranks), np.var(ranks)

def eval_rank_histogram(true_data, gen_data, var, methods):
    axis_ranks = -1
    ground_truth = np.mean(true_data[var], axis = (-2, -1))
    means = []
    variances = []
    rel_error_means = []
    rel_error_variances = []
    for method in methods:
        forecasts = np.mean(gen_data[method][var], axis = (-3, -2))
        hist, mean, variance = compute_rank_histogram(ground_truth, forecasts, axis = axis_ranks, method = "min")
        mean_truth = (forecasts.shape[axis_ranks] + 1) / 2
        variance_truth = ((forecasts.shape[axis_ranks] + 1)**2 - 1)/12
        means.append(mean)
        variances.append(variance)
        rel_error_means.append((mean - mean_truth) / mean_truth)
        rel_error_variances.append((variance / variance_truth) - 1)
    
    return pd.DataFrame({"rh-mean": means, "rh-variance": variances, "rh-rel_error_mean": rel_error_means, "rh-rel_error_variance": rel_error_variances}, index=methods)

def eval_rank_histogram_agg(ground_truth, forecasts_methods, methods, full=True, prefix=""):
    # RH function without aggregation internally
    axis_ranks = -1
    #ground_truth = np.mean(true_data[var], axis = (-2, -1))
    if full:
        means = []
        variances = []
        rel_error_means = []
        rel_error_variances = []
    mcb = []
    mcb_first_bin = []
    mcb_last_bin = []
    mcb_first_bin_signed = []
    mcb_last_bin_signed = []
    hists = {}
    for method in methods:
        forecasts = forecasts_methods[method] # np.mean(gen_data[method][var], axis = (-3, -2))
        hist, mean, variance = compute_rank_histogram_efficient(ground_truth, forecasts, axis = axis_ranks, method = "randomized")
        hists[method] = hist[0]
        if full:
            mean_truth = (forecasts.shape[axis_ranks] + 1) / 2
            variance_truth = ((forecasts.shape[axis_ranks] + 1)**2 - 1)/12
            means.append(mean)
            variances.append(variance)
            rel_error_means.append((mean - mean_truth) / mean_truth)
            rel_error_variances.append((variance / variance_truth) - 1)
        n_bins = forecasts.shape[axis_ranks] + 1
        mcb.append(np.sum(np.abs(hist[0] - 1/n_bins)))
        mcb_first_bin.append(np.abs(hist[0][0] - 1/n_bins))
        mcb_first_bin_signed.append(hist[0][0] - 1/n_bins)
        mcb_last_bin.append(np.abs(hist[0][-1] - 1/n_bins))
        mcb_last_bin_signed.append(hist[0][-1] - 1/n_bins)
    
    if full:
        return pd.DataFrame({f"{prefix}rh-mean": means, f"{prefix}rh-variance": variances, 
                             f"{prefix}rh-rel_error_mean": rel_error_means, f"{prefix}rh-rel_error_variance": rel_error_variances,
                         f"{prefix}mcb": mcb, 
                         f"{prefix}mcb-first-bin": mcb_first_bin, 
                         f"{prefix}mcb-last-bin": mcb_last_bin,
                         f"{prefix}mcb-first-bin-signed": mcb_first_bin_signed,
                         f"{prefix}mcb-last-bin-signed": mcb_last_bin_signed
                         }, index=methods), hists
    else:
        return pd.DataFrame({f"{prefix}mcb": mcb}, index=methods)

def mcb(hist):
    n_bins = len(hist)
    return np.sum(np.abs(hist - 1/n_bins))


def eval_rh_perloc(true_data, gen_data, var = "tas", methods = None,
                             num_samples=9, num_locs = 100):
    
    mcb_locs = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    means_locs = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    var_locs = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    mcb_last_locs = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    mcb_first_locs = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    mcb_last_locs_signed = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}
    mcb_first_locs_signed = {method: np.empty((true_data[var].shape[-2], true_data[var].shape[-1])) for method in methods}

    possible_indices = [(i, j) for i in range(true_data[var].shape[-2]) for j in range(true_data[var].shape[-1])]
    if num_locs < len(possible_indices):
        selected_indices = np.random.choice(len(possible_indices), num_locs, replace=False)
        selected_indices = [possible_indices[idx] for idx in selected_indices]
    else:
        selected_indices = possible_indices
    for i in range(true_data[var].shape[-2]):
        for j in range(true_data[var].shape[-1]):
                ground_truth_loc = true_data[var][..., i, j]                               
                forecasts_loc = {method: gen_data[method][var][..., i, j, :num_samples] for method in methods}

                if (i, j) not in selected_indices:
                    for method in methods:
                        mcb_locs[method][i, j] = 0
                        var_locs[method][i, j] = 0
                        means_locs[method][i, j] = 0
                    continue
                rh_diff_per_loc, _ = eval_rank_histogram_agg(ground_truth_loc, forecasts_loc, methods = methods, full=True)
                for method in methods:
                    mcb_locs[method][i, j] = rh_diff_per_loc.loc[method, "mcb"]
                    means_locs[method][i, j] = rh_diff_per_loc.loc[method, "rh-mean"]
                    var_locs[method][i, j] = rh_diff_per_loc.loc[method, "rh-variance"]
                    mcb_last_locs[method][i, j] = rh_diff_per_loc.loc[method, "mcb-last-bin"]
                    mcb_first_locs[method][i, j] = rh_diff_per_loc.loc[method, "mcb-first-bin"]
                    mcb_last_locs_signed[method][i, j] = rh_diff_per_loc.loc[method, "mcb-last-bin-signed"]
                    mcb_first_locs_signed[method][i, j] = rh_diff_per_loc.loc[method, "mcb-first-bin-signed"]
                
    return mcb_locs, means_locs, var_locs, mcb_last_locs, mcb_first_locs, mcb_last_locs_signed, mcb_first_locs_signed


def eval_fss(true_data, gen_data, var, methods, quantiles = [0.5, 0.6, 0.7], window_size=10):
    res = {}
    thresholds = {}
        
    for q in quantiles:
        res[q] = []
        thresholds[q] = np.quantile(true_data[var], q)

        mask = np.max(true_data[var], axis=(1,2)) > thresholds[q]
        true_filtered = true_data[var][mask]

        for method in methods:
            gen_filtered = gen_data[method][var][mask, :, :, 0]
            
            res[q].append(np.mean([pysteps.verification.spatialscores.fss(true_filtered[i, :, :], gen_filtered[i, :, :],
                                                                            thr = thresholds[q], 
                                                                            scale = window_size) for i in range(len(true_filtered))]))
                
    results_df = pd.DataFrame({f'FSS-thresh{q}': res[q] for q in quantiles}, index=methods)
    return results_df

def quantile_loss(q, y_hat, y):
    return (1-q) * (y_hat - y) * (y_hat > y) + q * (y - y_hat) * (y_hat <= y)

def eval_quantile_loss(true_data, gen_data, var, methods, pool_method = None, pool_quantile=0.9, quantiles = [0.1, 0.3, 0.5, 0.6, 0.9]):
    res_quantiles = {}
    for q in quantiles:
        if pool_method != "quantile":
            res_quantiles[f"QL-{pool_method}_q-{q}"] = []
        else:
            res_quantiles[f"QL-{pool_method}-qpool{pool_quantile}_q-{q}"] = []
    # Loop over the methods
    for method in methods:
        # Append the results to the dataframe
        if pool_method is None:
            gen = gen_data[method][var]
            true = true_data[var]
        elif pool_method == "mean":
            gen = np.mean(gen_data[method][var], axis=(1,2))
            true = np.mean(true_data[var], axis=(1,2))
        elif pool_method == "max":
            gen = np.max(gen_data[method][var], axis=(1,2))
            true = np.max(true_data[var], axis=(1,2))
        elif pool_method == "quantile":
            gen = np.quantile(gen_data[method][var], pool_quantile, axis=(1,2))
            true = np.quantile(true_data[var], pool_quantile, axis=(1,2))
        for q in quantiles:
            if pool_method != "quantile":
                res_quantiles[f"QL-{pool_method}_q-{q}"].append(quantile_loss(q, np.quantile(gen, q, axis=-1), true).mean())
            else:
                res_quantiles[f"QL-{pool_method}-qpool{pool_quantile}_q-{q}"].append(quantile_loss(q, np.quantile(gen, q, axis=-1), true).mean())
            
    results_df = pd.DataFrame(res_quantiles, index=methods)
    return results_df

def eval_quantile_loss_boot(true_data, gen_data, var, methods, pool_method = None, pool_quantile=0.9, quantiles = [0.1, 0.3, 0.5, 0.6, 0.9]):
    results = {f'QL-{pool_method}_q-{q}': {} for q in quantiles}
    if pool_method == "quantile":
        results = {f'QL-{pool_method}-qpool{pool_quantile}_q-{q}': {} for q in quantiles}

    # Loop over the methods
    for method in methods:
        if pool_method is None:
            gen = gen_data[method][var]
            true = true_data[var]
        elif pool_method == "mean":
            gen = np.mean(gen_data[method][var], axis=(1,2))
            true = np.mean(true_data[var], axis=(1,2))
        elif pool_method == "max":
            gen = np.max(gen_data[method][var], axis=(1,2))
            true = np.max(true_data[var], axis=(1,2))
        elif pool_method == "quantile":
            gen = np.quantile(gen_data[method][var], pool_quantile, axis=(1,2))
            true = np.quantile(true_data[var], pool_quantile, axis=(1,2))
        
        for q in quantiles:
            if pool_method is None:
                # average over space
                results[f"QL-{pool_method}_q-{q}"][method] = quantile_loss(q, np.quantile(gen, q, axis=-1), true).mean(axis = (1,2))
            elif pool_method == "mean" or pool_method == "max":
                results[f"QL-{pool_method}_q-{q}"][method] = quantile_loss(q, np.quantile(gen, q, axis=-1), true)
            else:
                results[f"QL-{pool_method}-qpool{pool_quantile}_q-{q}"][method] = quantile_loss(q, np.quantile(gen, q, axis=-1), true)
    
    return results

def eval_quantiles_marginal(true_data, gen_data, var, methods, quantiles = [0.95], months=None, time_vals_months=None):
    
    true = true_data[var]
    if months is not None:            
        true = true[np.isin(time_vals_months, months)]
        
    q_true = {}
    for q in quantiles:
        q_true[q] = np.quantile(true, q, axis=0)
    
    months_str = "-".join([str(month) for month in months])
    avg_abs_errors = {f"q-error_months-{months_str}_q-{q}": [] for q in quantiles}
    
    for method in methods:
        gen = gen_data[method][var]
        if months is not None:            
            gen = gen[np.isin(time_vals_months, months)]
        
        for q in quantiles:
            avg_abs_error = np.mean(np.abs(np.quantile(gen, q, axis=(0, -1)) - q_true[q]))
            avg_abs_errors[f"q-error_months-{months_str}_q-{q}"].append(avg_abs_error)
    
    results_df = pd.DataFrame(avg_abs_errors, index=methods)
    return results_df

def eval_quantile_loss_from_prep(true_data, gen_data_q, var, methods, quantiles = [0.1, 0.3, 0.5, 0.6, 0.9], pool_method=None):
    # same as eval_quantile_loss, but from precomputed quantiles
    res_quantiles = {}
    for q in quantiles:
        res_quantiles[f"QL-{pool_method}_q-{q}"] = []

    # Loop over the methods
    for method in methods:
        # Append the results to the dataframe
        gen = gen_data_q[method][var]
        true = true_data[var]
        for q in quantiles:
            res_quantiles[f"QL-{pool_method}_q-{q}"].append(quantile_loss(q, gen, true).mean())
            
    results_df = pd.DataFrame(res_quantiles, index=methods)
    return results_df

def eval_quantile_loss_from_prep_boot(true_data, gen_data_q, var, methods, quantiles = [0.1, 0.3, 0.5, 0.6, 0.9], pool_method=None):
    # same as eval_quantile_loss_from_prep, but no average over the days, allows for more efficient bootstrap
    results = {f"QL-{pool_method}_q-{q}": {} for q in quantiles}
    
    # Loop over the methods
    for method in methods:
        # Append the results to the dataframe
        gen = gen_data_q[method][var]
        true = true_data[var]
        for q in quantiles:
            ql = quantile_loss(q, gen, true)
            if len(ql.shape) == 3:
                ql = ql.mean(axis=(1,2))
            results[f"QL-{pool_method}_q-{q}"][method] = ql
            
    return results

def eval_maxima_loc(true_data, gen_data, var = "tas", methods = None, weighted = False):
    # create a list to store the results    
    
    average_distances = []
    # iterate over each method
    for method in methods:
        
        distances = []
        for i in range(len(true_data[var])):
            for j in range(gen_data[method][var].shape[-1]):
                array = gen_data[method][var][i, :, :, j]
                gen_max = np.max(array)
                row, col = np.unravel_index(np.argmax(array), array.shape)
                true_at_genmax = true_data[var][i, row, col]

                array = true_data[var][i, :, :]
                true_max = np.max(array)
                row_t, col_t = np.unravel_index(np.argmax(array), array.shape)
                gen_at_truemax = gen_data[method][var][i, row_t, col_t, j]

                if weighted:
                    distance = np.sqrt((row - row_t)**2 + (col - col_t)**2) * np.abs(gen_at_truemax - true_max) * np.abs(true_at_genmax - gen_max)
                else:
                    distance = np.sqrt((row - row_t)**2 + (col - col_t)**2)
                distances.append(distance)

        average_distance = np.mean(distances)
        
        # append the scores to the dataframe
        average_distances.append(average_distance)
        
    # create a dataframe
    df = pd.DataFrame({f'dist-max_w-{weighted}': average_distances}, index=methods)
    return df
    
def eval_psd_boot(true_data, gen_data, var="tas", methods=None, n_samples=3, verbose=False):
    results = {"LSD": {}}

    # Compute mean PSD over true data
    rapsd_true = np.stack([pysteps.utils.spectral.rapsd(true_data[var][i, :, :], fft_method=np.fft) 
        for i in range(true_data[var].shape[0])], axis = 0)
    
    log_rapsd_true = np.log10(rapsd_true + 1e-8)  # add epsilon to avoid log(0)
    
    # Iterate over each method
    rapsd_gen_dict = {}
    for method in methods:
        data = gen_data[method][var]
        n_ens = data.shape[-1]
        used_samples = min(n_samples, n_ens)

        # Compute mean PSD across ensemble members
        psd_mat = np.empty((data.shape[0],rapsd_true.shape[-1], used_samples))
        for i in range(data.shape[0]):  # over time steps
            for j in range(used_samples):  # over ensemble members
                psd_mat[i, :, j] = pysteps.utils.spectral.rapsd(data[i, :, :, j], fft_method=np.fft)

        log_rapsd_gen = np.log10(psd_mat + 1e-8)  # avoid log(0)

        # Compute RMSE in log spectral space
        # average over 3 samples
        lsd_per_sample = np.stack([np.sqrt(np.nanmean(100 * (log_rapsd_true - log_rapsd_gen[..., j]) ** 2, axis = 1))
               for j in range(used_samples)], axis=-1)
        lsd = np.mean(lsd_per_sample, axis=-1)  # average over samples
        
        rapsd_gen = np.nanmean(psd_mat, axis=-1)
        rapsd_gen_dict[method] = rapsd_gen
        results["LSD"][method] = lsd

    if verbose:
        return results, rapsd_true, rapsd_gen_dict
    else:
        return results


def eval_psd_boot_noagg(true_data, gen_data, var="tas", methods=None, n_samples=3):

    # Compute mean PSD over true data
    rapsd_true = np.stack([pysteps.utils.spectral.rapsd(true_data[var][i, :, :], fft_method=np.fft) 
        for i in range(true_data[var].shape[0])], axis = 0)
        
    # Iterate over each method
    rapsd_gen_dict = {}
    for method in methods:
        data = gen_data[method][var]
        n_ens = data.shape[-1]
        used_samples = min(n_samples, n_ens)

        # Compute mean PSD across ensemble members
        psd_mat = np.empty((data.shape[0],rapsd_true.shape[-1], used_samples))
        for i in range(data.shape[0]):  # over time steps
            for j in range(used_samples):  # over ensemble members
                psd_mat[i, :, j] = pysteps.utils.spectral.rapsd(data[i, :, :, j], fft_method=np.fft)
        
        #rapsd_gen = np.nanmean(psd_mat, axis=-1)
        rapsd_gen_dict[method] = psd_mat

    return rapsd_true, rapsd_gen_dict


def bootstrap_eval_eff_psd(true_data, gen_data, var="tas", methods=None, n_bootstrap=100, sample_frac=1.0, **eval_kwargs):
    """
    Wrapper to calculate bootstrap estimates for any evaluation function with optional subsampling.
    
    Parameters:
        eval_func (callable): Evaluation function to bootstrap (e.g., eval_mse, eval_es).
        true_data (dict): Dictionary of true data arrays.
        gen_data (dict): Dictionary of generated data arrays for different methods.
        var (str): Variable name to evaluate.
        methods (list): List of method names to evaluate.
        n_bootstrap (int): Number of bootstrap resamples.
        sample_frac (float): Fraction of the dataset to use in each bootstrap resample (0 < sample_frac ≤ 1).
        eval_kwargs (dict): Additional keyword arguments for the evaluation function.
    
    Returns:
        pd.DataFrame: DataFrame containing mean and standard error for each metric.
    """
    n_samples = int(true_data[var].shape[0] * sample_frac)
    if n_samples < 1:
        raise ValueError("sample_frac is too small; results in fewer than 1 sample.")
    
    if methods is None:
        methods = gen_data.keys()
    rapsd_true, rapsd_gen_dict = eval_psd_boot_noagg(true_data, gen_data, var=var, methods=methods, **eval_kwargs)
    aggregated_results = {}
    for score in ["LSD-daily", "LSD-avg"]:
        bootstrap_results = {method: [] for method in methods}
        for method in methods:
            # Perform bootstrap resampling
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(rapsd_gen_dict[method]), n_samples, replace=True)
                if score == "LSD-avg":
                    true_psd = rapsd_true[indices].mean(axis = 0)
                    gen_psd = rapsd_gen_dict[method][indices].mean(axis = (0, -1))
                    
                    # Compute RMSE in log spectral space; for PSD averaged across days
                    lsd = np.sqrt(np.nanmean(100 * (np.log10(true_psd) - np.log10(gen_psd)) ** 2, axis = 0))
                elif score == "LSD-daily":
                    true_psd = rapsd_true[indices]
                    gen_psd = rapsd_gen_dict[method][indices]

                    # Compute RMSE in log spectral space; this time for single days, then average over days
                    lsd_per_sample = np.stack([np.sqrt(np.nanmean(100 * (np.log10(true_psd) - np.log10(gen_psd[..., j])) ** 2, axis = 1))
                        for j in range(gen_psd.shape[-1])], axis=-1)
                    lsd = np.mean(lsd_per_sample, axis=(0,-1))
                
                bootstrap_results[method].append(lsd)
        
            aggregated_results[f"mean*{score}"] = [np.mean(bootstrap_results[method]) for method in methods]
            aggregated_results[f"se*{score}"] = [np.std(bootstrap_results[method]) for method in methods]
    
    combined_results = pd.DataFrame(aggregated_results, index=methods)
    return combined_results, rapsd_true, rapsd_gen_dict



def acf_lag1(x):
    """Compute autocorrelation for lag 1 efficiently."""
    x = np.asarray(x)
    mean_x = np.mean(x)
    var_x = np.var(x)
    return np.sum((x[:-1] - mean_x) * (x[1:] - mean_x)) / (len(x) * var_x)


def eval_acf(true_data, gen_data, var="tas", methods=None, pool_method=None, kernel_size=1):
    acf_errors = []
    acf_diffs = []
    if methods is None:
        methods = gen_data.keys()
    
    for method in methods:
        true = torch.tensor(true_data[var].copy()).unsqueeze(1)
        gen = torch.tensor(gen_data[method][var][..., 0].copy()).unsqueeze(1)
        
        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen)
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen)
        
        true_np = true.squeeze().numpy()
        gen_np = gen.squeeze().numpy()
        
        acf_lag1_true = np.apply_along_axis(acf_lag1, 0, true_np)
        acf_lag1_gen = np.apply_along_axis(acf_lag1, 0, gen_np)
        
        abs_error = np.mean(np.abs(acf_lag1_true - acf_lag1_gen))  # Absolute error averaged over all locations
        acf_diff = np.mean(acf_lag1_gen - acf_lag1_true) # Mean difference in ACF lag 1
        acf_diffs.append(acf_diff)
        acf_errors.append(abs_error)
    
    df = pd.DataFrame({f'ACF_lag1_abs_error_pool-{pool_method}_kernel-{kernel_size}': acf_errors,
                       f'ACF_lag1_diff_pool-{pool_method}_kernel-{kernel_size}': acf_diffs
                       }, index=methods)
    return df

def eval_acf_spatial(true_data, gen_data, var="tas", methods=None, pool_method=None, kernel_size=1):

    acf_errors = {}    
    acf_gens = {}
    if methods is None:
        methods = gen_data.keys()
    
    for method in methods:
        true = torch.tensor(true_data[var].copy()).unsqueeze(1)
        gen = torch.tensor(gen_data[method][var][..., 0].copy()).unsqueeze(1)
        
        if pool_method == "average":
            pool = torch.nn.AvgPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen)
        elif pool_method == "max":
            pool = torch.nn.MaxPool2d(kernel_size)
            true = pool(true)
            gen = pool(gen)
        
        true_np = true.squeeze().numpy()
        gen_np = gen.squeeze().numpy()
        
        acf_lag1_true = np.apply_along_axis(acf_lag1, 0, true_np)
        acf_lag1_gen = np.apply_along_axis(acf_lag1, 0, gen_np)
        acf_gens[method] = acf_lag1_gen
        
        abs_error = np.abs(acf_lag1_true - acf_lag1_gen)
        acf_errors[method] = abs_error
    
    return acf_lag1_true, acf_gens, acf_errors

def eval_pointwise_corr(true_data, gen_data, var1 = "tas", var2 = "pr", 
                        methods = None, pool_method=None, kernel_size=1, j=0):
    corrs = []
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

        corrs.append(np.mean(np.abs(correlation_array_gen - correlation_array_true)))
        
    # create a dataframe
    df = pd.DataFrame({f'Corr_pool-{pool_method}_kernel-{kernel_size}': corrs}, index=methods)
    return df

# joint in variables, and conditional in time
def eval_pw_exceedance_joint_conditional(true_data, gen_data, var1 = "tas", var2 = "pr", 
                        methods=None, q1=0.9, q2 = 0.3, mode1= "ex", mode2="sub",
                        month_array=None, months=np.arange(1,13), return_individual_months=True):

    if methods is None:
        methods = gen_data.keys()

    monthly_dfs = {}
    for month in months:
        monthly_dfs[month] = []
        mask = (month_array == month)
        true1 = true_data[var1][mask, :, :]
        true2 = true_data[var2][mask, :, :]
        
        local_q1 = np.quantile(true1, q=q1, axis=0)
        local_q2 = np.quantile(true2, q=q2, axis=0)
                
        if mode1 == "ex":
            true_ex1 = (true1 > local_q1)
        elif mode1 == "sub":
            true_ex1 = (true1 < local_q1)
        if mode2 == "ex":
            true_ex2 = (true2 > local_q2)
        elif mode2 == "sub":
            true_ex2 = (true2 < local_q2)
        true_joint_ex = np.logical_and(true_ex1, true_ex2)
        
        for method in methods:
            gen1 = gen_data[method][var1][mask, :, :, :]
            gen2 = gen_data[method][var2][mask, :, :, :]
                    
            # count exceedances along last axis
            if mode1 == "ex":
                gen_ex1 = np.stack([gen1[:,:,:,j] > local_q1 for j in range(gen1.shape[3])], axis=-1)
            elif mode1 == "sub":
                gen_ex1 = np.stack([gen1[:,:,:,j] < local_q1 for j in range(gen1.shape[3])], axis=-1)
            if mode2 == "ex":
                gen_ex2 = np.stack([gen2[:,:,:,j] > local_q2 for j in range(gen2.shape[3])], axis=-1)
            elif mode2 == "sub":
                gen_ex2 = np.stack([gen2[:,:,:,j] < local_q2 for j in range(gen2.shape[3])], axis=-1)
            
            gen_joint_ex_prob = np.mean(np.logical_and(gen_ex1, gen_ex2), axis=-1)
            monthly_dfs[month].append(ps.brier_score(true_joint_ex, gen_joint_ex_prob).mean())
            
    results_df = pd.DataFrame({f'Brier-m{month}_q{q1}-{q2}': monthly_dfs[month] for month in months}, index=methods)
    results_df[f"Brier-monthly-avg_q{q1}-{q2}"] = results_df.mean(axis=1)
    if return_individual_months:
        return results_df
    else:
        return results_df.iloc[:, -1]

# joint in variables, but marginal in time
def eval_pw_exceedance_joint_marginal(true_data, gen_data, var1 = "tas", var2 = "pr", 
                        methods=None, q1=0.9, q2 = 0.3, mode1= "ex", mode2="sub",
                        month_array=None, months=np.arange(1,13), return_individual_months=True):

    if methods is None:
        methods = gen_data.keys()

    monthly_dfs = {}
    for month in months:
        monthly_dfs[month] = []
        mask = (month_array == month)
        true1 = true_data[var1][mask, :, :]
        true2 = true_data[var2][mask, :, :]
        
        local_q1 = np.quantile(true1, q=q1, axis=0)
        local_q2 = np.quantile(true2, q=q2, axis=0)
                
        if mode1 == "ex":
            true_ex1 = (true1 > local_q1)
        elif mode1 == "sub":
            true_ex1 = (true1 < local_q1)
        if mode2 == "ex":
            true_ex2 = (true2 > local_q2)
        elif mode2 == "sub":
            true_ex2 = (true2 < local_q2)
        true_joint_ex = np.logical_and(true_ex1, true_ex2)
        true_joint_ex_prob = np.mean(true_joint_ex, axis=0)
        
        for method in methods:
            gen1 = gen_data[method][var1][mask, :, :, :] # changed to all axis
            gen2 = gen_data[method][var2][mask, :, :, :]
                    
            if mode1 == "ex":
                gen_ex1 = np.stack([gen1[:,:,:,j] > local_q1 for j in range(gen1.shape[3])], axis=-1)
            elif mode1 == "sub":
                gen_ex1 = np.stack([gen1[:,:,:,j] < local_q1 for j in range(gen1.shape[3])], axis=-1)
            if mode2 == "ex":
                gen_ex2 = np.stack([gen2[:,:,:,j] > local_q2 for j in range(gen2.shape[3])], axis=-1)
            elif mode2 == "sub":
                gen_ex2 = np.stack([gen2[:,:,:,j] < local_q2 for j in range(gen2.shape[3])], axis=-1)
            
            gen_joint_ex_prob = np.mean(np.logical_and(gen_ex1, gen_ex2), axis=(0,-1))
            monthly_dfs[month].append(np.mean(np.abs(gen_joint_ex_prob - true_joint_ex_prob)))
            
    results_df = pd.DataFrame({f'Joint-ex-prob-m{month}_q{q1}-{q2}': monthly_dfs[month] for month in months}, index=methods)
    results_df[f"Joint-ex-prob-monthly-avg_q{q1}-{q2}"] = results_df.mean(axis=1)
    if return_individual_months:
        return results_df
    else:
        return results_df.iloc[:, -1]


# marginal in variables, but conditional in time
def eval_pw_exceedance_marginal_conditional(true_data, gen_data, var1 = "tas",
                        methods=None, q1=0.9, mode1= "ex",
                        month_array=None, months=np.arange(1,13), return_individual_months=True):

    if methods is None:
        methods = gen_data.keys()

    monthly_dfs = {}
    for month in months:
        monthly_dfs[month] = []
        mask = (month_array == month)
        true1 = true_data[var1][mask, :, :]        
        local_q1 = np.quantile(true1, q=q1, axis=0)
                        
        if mode1 == "ex":
            true_ex1 = (true1 > local_q1)
        elif mode1 == "sub":
            true_ex1 = (true1 < local_q1)
        
        for method in methods:
            gen1 = gen_data[method][var1][mask, :, :, :] # changed to all axis
                    
            # count exceedances along last axis
            if mode1 == "ex":
                gen_ex1 = np.stack([gen1[:,:,:,j] > local_q1 for j in range(gen1.shape[3])], axis=-1)
            elif mode1 == "sub":
                gen_ex1 = np.stack([gen1[:,:,:,j] < local_q1 for j in range(gen1.shape[3])], axis=-1)
            
            gen_ex_prob = np.mean(gen_ex1, axis=-1)
            
            monthly_dfs[month].append(ps.brier_score(true_ex1, gen_ex_prob).mean())
            
    results_df = pd.DataFrame({f'Brier-m{month}_q{q1}': monthly_dfs[month] for month in months}, index=methods)
    results_df[f"Brier-monthly-avg_q{q1}"] = results_df.mean(axis=1)
    if return_individual_months:
        return results_df
    else:
        return results_df.iloc[:, -1]

# marginals in both variables and time
def eval_pw_exceedance_marginal_marginal(true_data, gen_data, var1 = "tas",
                        methods=None, q1=0.9,
                        month_array=None, months=np.arange(1,13), return_individual_months=True):

    if methods is None:
        methods = gen_data.keys()

    monthly_dfs = {}
    for month in months:
        monthly_dfs[month] = []
        mask = (month_array == month)
        true1 = true_data[var1][mask, :, :]        
        local_q1 = np.quantile(true1, q=q1, axis=0)
        
        for method in methods:
            gen1 = gen_data[method][var1][mask, :, :, :] # changed to all axis
                    
            # count exceedances along first and last axis
            gen_sub1 = np.stack([gen1[:,:,:,j] < local_q1 for j in range(gen1.shape[3])], axis=-1)
            gen_sub_prob = np.mean(gen_sub1, axis=(0,-1))
            monthly_dfs[month].append(np.mean(np.abs(gen_sub_prob - q1)))
            
    results_df = pd.DataFrame({f'Marginal-sub-prob-m{month}_q{q1}': monthly_dfs[month] for month in months}, index=methods)
    results_df[f"Marginal-sub-prob-monthly-avg_q{q1}"] = results_df.mean(axis=1)
    if return_individual_months:
        return results_df
    else:
        return results_df.iloc[:, -1]


def eval_rh_var_diffs_perloc(true_data, gen_data, var1 = "tas", var2 = "pr", methods = None,
                             primitive_standardization=False, num_samples=9, num_locs = 100):
    
    mcb_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}
    means_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}
    var_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods}

    possible_indices = [(i, j) for i in range(true_data[var1].shape[-2]) for j in range(true_data[var1].shape[-1])]
    selected_indices = np.random.choice(len(possible_indices), num_locs, replace=False)
    selected_indices = [possible_indices[idx] for idx in selected_indices]
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

# conditional correlations

def eval_conditional_corr(gen_data, var1 = "tas", var2 = "pr",
                          methods=None, month_array=None, months=np.arange(1,13), base_method = "analogues", return_individual_months=True):
    
    if methods is None:
        methods = gen_data.keys()

    monthly_dfs = {}
    for month in months:
        monthly_dfs[month] = []
        mask = (month_array == month)
        
        corrs_dict = {}
        for method in methods:
            gen1 = gen_data[method][var1][mask, :, :, :] # changed to all axis
            gen2 = gen_data[method][var2][mask, :, :, :]
            
            corrs = np.zeros((gen1.shape[0], gen1.shape[1], gen1.shape[2]))
            for t in range(gen1.shape[0]):
                for loc_i in range(gen1.shape[1]):
                    for loc_j in range(gen1.shape[2]):
                        corrs[t, loc_i, loc_j] = np.corrcoef(gen1[t, loc_i, loc_j, :], gen2[t, loc_i, loc_j, :])[0, 1]
            
            corrs_dict[method] = np.mean(corrs, axis=0)
    
        for method in methods:
            monthly_dfs[month].append(np.mean(np.abs(corrs_dict[method] - corrs_dict[base_method])))
            
    results_df = pd.DataFrame({f'Corr-m{month}': monthly_dfs[month] for month in months}, index=methods)
    results_df[f"Corr-monthly-avg"] = results_df.mean(axis=1)
    
    if return_individual_months:
        return results_df
    else:
        return results_df.iloc[:, -1]      
                            
def ecdf_full(data, flatten = False):
    if flatten:
        len_data = len(data)
        data = data.flatten()
    ranks = rankdata(data, method='average')
    ecdf = ranks / (len(data) + 1)                        
    if flatten:
        ecdf = ecdf.reshape(len_data, -1)
    return ecdf

def dirty_eval_all(true_data, gen_data, var, methods):
    return pd.concat([eval_es(true_data, gen_data, var, methods = methods, pool_method=None),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=10),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=10),
    eval_mse(true_data, gen_data, var, methods = methods),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method=None),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=10),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=10),
    eval_fss(true_data, gen_data, var, methods = methods, quantiles = [0.5, 0.9], window_size=10),
    #eval_rank_histogram(true_data, gen_data, var, methods = methods)
    eval_quantile_loss(true_data, gen_data, var, methods = methods, pool_method=None),
    eval_quantile_loss(true_data, gen_data, var, methods = methods, pool_method="mean"),
    eval_quantile_loss(true_data, gen_data, var, methods = methods, pool_method="max"),
    eval_quantile_loss(true_data, gen_data, var, methods = methods, pool_method="quantile-0.9")],
              axis = 1)
    
def dirty_eval_subset1(true_data, gen_data, var, methods):
    return pd.concat([eval_es(true_data, gen_data, var, methods = methods, pool_method=None),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=2),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=2),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=10),
    eval_es(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=10),
    eval_mse(true_data, gen_data, var, methods = methods)],
              axis = 1)
    
def dirty_eval_subset2(true_data, gen_data, var, methods):
    return pd.concat([eval_crps(true_data, gen_data, var, methods = methods, pool_method=None),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=2),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=2),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="average", kernel_size=10),
    eval_crps(true_data, gen_data, var, methods = methods, pool_method="max", kernel_size=10)
    ],
              axis = 1)
    
### ------ BOOTSTRAP WRAPPER ----------------    
    
def bootstrap_eval(eval_func, true_data, gen_data, var="tas", methods=None, n_bootstrap=100, sample_frac=1.0, **eval_kwargs):
    """
    Wrapper to calculate bootstrap estimates for any evaluation function with optional subsampling.
    
    Parameters:
        eval_func (callable): Evaluation function to bootstrap (e.g., eval_mse, eval_es).
        true_data (dict): Dictionary of true data arrays.
        gen_data (dict): Dictionary of generated data arrays for different methods.
        var (str): Variable name to evaluate.
        methods (list): List of method names to evaluate.
        n_bootstrap (int): Number of bootstrap resamples.
        sample_frac (float): Fraction of the dataset to use in each bootstrap resample (0 < sample_frac ≤ 1).
        eval_kwargs (dict): Additional keyword arguments for the evaluation function.
    
    Returns:
        pd.DataFrame: DataFrame containing mean and standard error for each metric.
    """
    n_samples = int(true_data[var].shape[0] * sample_frac)
    if n_samples < 1:
        raise ValueError("sample_frac is too small; results in fewer than 1 sample.")
    
    bootstrap_results = []

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        indices = np.random.choice(true_data[var].shape[0], n_samples, replace=True)
        true_resampled = {var: true_data[var][indices, ...]}
        gen_resampled = {method: {var: gen_data[method][var][indices, ...]} for method in methods}
        result = eval_func(true_resampled, gen_resampled, var=var, methods=methods, **eval_kwargs)
        bootstrap_results.append(result)

    bootstrap_concat = pd.concat(bootstrap_results, axis=0)
    mean_results = bootstrap_concat.groupby(bootstrap_concat.index).mean()
    se_results = bootstrap_concat.groupby(bootstrap_concat.index).std() * np.sqrt(sample_frac) # / np.sqrt(n_bootstrap) #* n_samples)

    # Combine mean and standard error into a single DataFrame
    combined_results = pd.concat([mean_results, se_results], axis=1, keys=["mean", "se"])
    combined_results.columns = ['*'.join(col).strip() for col in combined_results.columns.values]
    return combined_results



def bootstrap_eval_eff(eval_func, true_data, gen_data, var="tas", methods=None, n_bootstrap=100, sample_frac=1.0, **eval_kwargs):
    """
    Wrapper to calculate bootstrap estimates for any evaluation function with optional subsampling.
    
    Parameters:
        eval_func (callable): Evaluation function to bootstrap (e.g., eval_mse, eval_es).
        true_data (dict): Dictionary of true data arrays.
        gen_data (dict): Dictionary of generated data arrays for different methods.
        var (str): Variable name to evaluate.
        methods (list): List of method names to evaluate.
        n_bootstrap (int): Number of bootstrap resamples.
        sample_frac (float): Fraction of the dataset to use in each bootstrap resample (0 < sample_frac ≤ 1).
        eval_kwargs (dict): Additional keyword arguments for the evaluation function.
    
    Returns:
        pd.DataFrame: DataFrame containing mean and standard error for each metric.
    """
    n_samples = int(true_data[var].shape[0] * sample_frac)
    if n_samples < 1:
        raise ValueError("sample_frac is too small; results in fewer than 1 sample.")
    
    if methods is None:
        methods = gen_data.keys()
    result_full = eval_func(true_data, gen_data, var=var, methods=methods, **eval_kwargs)
    aggregated_results = {}
    for score in result_full.keys():
        bootstrap_results = {method: [] for method in methods}
        for method in methods:
            # Perform bootstrap resampling
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(result_full[score][method]), n_samples, replace=True)
                metric = result_full[score][method][indices].mean()
                bootstrap_results[method].append(metric)
        
        aggregated_results[f"mean*{score}"] = [np.mean(bootstrap_results[method]) for method in methods]
        aggregated_results[f"se*{score}"] = [np.std(bootstrap_results[method]) for method in methods]
    
    combined_results = pd.DataFrame(aggregated_results, index=methods)
    return combined_results