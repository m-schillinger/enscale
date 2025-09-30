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
import re
from utils import *
#
from eval_metrics_funs import *
from eval_funs import cond_energy_score_batch, cond_energy_score

import pysteps.utils.spectral

import argparse
from IPython import embed
import pdb
from collections import defaultdict
import multiprocessing as mp
################################################################################

def quantile_loss(q, y_hat, y):
    return (1-q) * (y_hat - y) * (y_hat > y) + q * (y - y_hat) * (y_hat <= y)

def file_log(fname, log_line, print_line=True):

    if print_line:
        print(log_line)

    with open(fname, 'a') as f:
        print(log_line, file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # create a global lock
    print_lock = mp.Lock()

    parser.add_argument('--project_specifier', type=str,
                        default='maybritt',
                        choices=['speed2zero', 'deepdown', 'maybritt'],
                        help="Specify project.")

    parser.add_argument('--subfolder' , type=str,
                        default = "maybritt",
                        choices = ["maybritt", "benchmarks"])

    parser.add_argument('--save_name', type=str,
                        default='',
                        help="Specify more details for the main folder.")

    parser.add_argument('--num_samples', type=int,
                        default=9,
                        help="Number of samples to use in plots and calculations")

    parser.add_argument('--run_id', '-i', type=str, nargs='+',
                        default=["maybritt_3step_dense_v1"],
                        help="Run Id to identify a training run")

    parser.add_argument('--pred_id', type=str,
                        default='002',
                        help="Id for different prediction settings (i.e. other checkpoints used).")
    
    parser.add_argument('--variables', type=str,
                        nargs='+', default=['tas', 'pr'])
    #
    # parser.add_argument('--data_path', type=str,
    #                     default='/mydata/speed2zero/shared/downscaling/cordex/cordex-ALPS-allyear',
    #                     help="Path to the data directory.")
    #
    # parser.add_argument('--predictions_path', type=str,
    #                     default='/mydata/speed2zero/shared/downscaling/tmp/predictions',
    #                     help="Path to the data directory.")
    #
    # parser.add_argument('--out_path', type=str,
    #                     default='output_maxim',
    #                     help="Path to the output directory.")

    # parser.add_argument('--ckpt_id', '-i', type=str,
    #                     default='192000',
    #                     help="Id of the particular checkpoint of training run.")

    parser.add_argument('--period', type=str,
                        default='2030-2039',
                        choices=['train-period', 'test-period', 'val-period', 'full-period',
                                 '2030-2039', '2090-2099'],
                        help="Period to calculate the statistics for.")

    parser.add_argument('--grid_type', type=str,
                        default='cordexgrid',
                        choices=['regular-g0d11', 'cordexgrid', 'dd'],
                        help="Specify the used grid.")

    parser.add_argument('--gcm', type=str,
                        default='all',
                        choices=['trained_on', 'all', 'CNRM-CM5', 'MPI-ESM-LR', 'MIROC5', 'None'],
                        help="Specify which GCM predictions are to be considered. "
                             "'trained_on' takes the GCM that was used for training.")

    parser.add_argument('--rcm', type=str,
                        default='all',
                        choices=['trained_on', 'all', 'ALADIN63', 'CCLM4-8-17', 'REMO2015', 'RegCM4-6', 'None'],
                        help="Specify which RCM predictions are to be considered."
                             "'trained_on' takes the RCM that was used for training.")

    parser.add_argument('--random_input', action='store_true',
                        help='Use 100 random inputs to study the prediction.')
    
    parser.add_argument('--num_bootstrap', type=int,
                        default=1, help='Number of bootstrap samples to compute SE.')

    parser.add_argument('--run_all', '-0', action='store_true',
                        help='Run all calculations and plotting steps.')
    
    parser.add_argument('--not_load_benchmarks', action='store_true',
                        help='Do not load benchmarks.')
    
    parser.add_argument('--not_load_gan', action='store_true',
                        help='Do not load GAN.')
    
    parser.add_argument('--not_load_diffusion', action='store_true',
                        help='Do not load diiffusion model.')

    parser.add_argument('--plot_pred', '-1', action='store_true',
                        help='Plots predictions.')

    parser.add_argument('--calc_mse', '-2', action='store_true',
                        help='Calculate MSE.')

    parser.add_argument('--calc_es', '-3', action='store_true',
                        help='Calculate energy scores.')

    parser.add_argument('--plot_quantiles', '-4', action='store_true',
                        help='Plot pixel-wise quantiles.')

    parser.add_argument('--plot_seasonality', '-5', action='store_true',
                        help='Plot monthly mean values.')

    parser.add_argument('--plot_seasonal_es', '-6', action='store_true',
                        help='Plot monthly mean energy score values.')

    parser.add_argument('--calc_crps', '-7', action='store_true',
                        help='Calculate CRPS.')

    parser.add_argument('--plot_ranks', '-8', action='store_true',
                        help='Plot rank histograms.')

    parser.add_argument('--calc_quantile_loss', '-9', action='store_true',
                        help='Calculate quantile losses.')

    parser.add_argument('--calc_spatial_metrics', '-10', action='store_true',
                        help='Calculate spatial metrics.')

    parser.add_argument('--calc_acf', '-11', action='store_true',
                        help='Calculate autocorrelations.')
    
    parser.add_argument('--plot_power_spectrum', '-12', action='store_true',
                        help='Plot power spectra.')

    parser.add_argument('--calc_corrs', '-13', action='store_true',
                        help='Calculate pixelwise pairwise correlations between variables.')
    
    parser.add_argument('--calc_joint_ex', '-14', action='store_true',
                        help='Calculate pixelwise joint exceedances between variables.')
    
    parser.add_argument('--rh_var_diffs', '-15', action='store_true',
                        help='RH summary stats for distance of standardised variables.')
    
    parser.add_argument('--calc_es_copula_vars', '-16', action='store_true',
                        help='ES on copula between variables')
    
    parser.add_argument('--calc_quantiles', '-17', action='store_true',
                        help = "marginal quantiles")
    
    parser.add_argument('--filter_outliers', action='store_true',
                        help='Filter outliers in the generated samples.')

    args = parser.parse_args()


    if args.project_specifier == 'deepdown':
        if args.run_all:
            args.run_all = False
            args.plot_pred = True
            args.calc_mse = True
            args.calc_es = True
            args.plot_quantiles = True
            args.plot_seasonality = True
            args.calc_crps = True
            args.plot_ranks = True
            args.calc_quantile_loss = True
            args.calc_spatial_metrics = False
            args.calc_fss = False
            args.calc_corrs = True
            args.calc_joint_ex = True
            args.rh_var_diffs = True

    if args.project_specifier == 'speed2zero':
        data_path = '/mydata/speed2zero/shared/downscaling/cordex/cordex-ALPS-allyear'
        predictions_path = '/mydata/speed2zero/shared/downscaling/tmp/predictions'
        out_path = 'output_maxim/speed2zero'
    elif args.project_specifier == 'deepdown':
        data_path = '/mydata/speed2zero/shared/DeepDown/MCH'
        predictions_path = '/mydata/speed2zero/shared/DeepDown/tmp/predictions'
        out_path = 'output_maxim/deepdown'
    elif args.project_specifier == 'maybritt':
        data_path = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
        if args.period == '2030-2039':
            subfolder_period = 'interpolation'
        elif args.period == '2090-2099':
            subfolder_period = 'extrapolation'
        else:
            subfolder_period = ''
        out_path = os.path.join('output_evals', args.subfolder, subfolder_period, args.save_name)
    else:
        raise ValueError("Check project specifier!")

    os.makedirs(out_path, exist_ok=True)
    if args.project_specifier != 'maybritt':
        out_path = os.path.join(out_path, f'pred_id-{args.pred_id}')
    os.makedirs(out_path, exist_ok=True)

    run_info = {
        # cordexgrid:
        # single-target: tas, pr separately:
        '290724_095300': {'content_loss': 'ensmeanMSE', 'target': 'pr', 'train_GCM': 'all', 'train_RCM': 'all'},
        '290724_101900': {'content_loss': 'ensmeanMSE', 'target': 'tas', 'train_GCM': 'all', 'train_RCM': 'all'},
        '290724_103600': {'content_loss': 'CRPS', 'target': 'tas', 'train_GCM': 'all', 'train_RCM': 'all'},
        '290724_111300': {'content_loss': 'CRPS', 'target': 'pr', 'train_GCM': 'all', 'train_RCM': 'all'},
        # multi-target: tas + pr:
        '290824_185100': {'content_loss': 'ensmeanMSE', 'target': 'all', 'train_GCM': 'all', 'train_RCM': 'all'},
        '290824_200400': {'content_loss': 'CRPS', 'target': 'all', 'train_GCM': 'all', 'train_RCM': 'all'},
    }
    
    if len(args.run_id) > 1:
        run_id_name = "multiple-methods"
    elif len(args.run_id) == 1:
        run_id_name = args.run_id[0]
    else:
        raise NotImplementedError("Multiple run_ids only implemented for maybritt_3step.")

    if args.project_specifier == 'maybritt':
        variables = args.variables
        content_loss = ""
    else:
        if run_info[args.run_id]['target'] == 'all':
            if args.project_specifier == 'speed2zero':
                variables = ['tas', 'pr']
            elif args.project_specifier == 'deepdown':
                variables = ['t', 'tp']
        else:
            variables = [run_info[args.run_id]['target']]
        content_loss = run_info[args.run_id]['content_loss']
        

    if args.grid_type == 'dd':
        gcm_rcm_dict = {None: [None]}
    elif args.grid_type == 'cordexgrid':
        gcm_rcm_dict = {'CNRM-CM5': ['ALADIN63', 'CCLM4-8-17', 'RegCM4-6'],
                        'MPI-ESM-LR': ['ALADIN63', 'CCLM4-8-17', 'RegCM4-6'],
                        'MIROC5': ['CCLM4-8-17', 'REMO2015']
                        }
    elif args.grid_type == 'regular-g0d11':
        gcm_rcm_dict = {'CNRM-CM5': ['ALADIN63', 'RegCM4-6'],
                        'MPI-ESM-LR': ['ALADIN63', 'CCLM4-8-17', 'RegCM4-6'],
                        'MIROC5': ['CCLM4-8-17', 'REMO2015']
                        }

    if args.gcm.startswith("trained_on"):
        args.gcm = run_info[args.run_id]['train_GCM']

    if args.rcm.startswith("trained_on"):
        args.rcm = run_info[args.run_id]['train_RCM']

    if args.gcm == 'None':
        args.gcm = None

    if args.rcm == 'None':
        args.rcm = None

    months = np.arange(1, 13)

    out_path_list = {}
    log_file_list = {}

    overview_mse_file_list = {}
    overview_es_file_list = {}
    overview_es_avg_file_list = {}
    overview_crps_no_pool_file_list = {}
    overview_crps_max_pool_file_list = {}
    overview_crps_window_pool_file_list = {}
    overview_ranks_file_list = {}
    overview_quantile_loss_file_list = {}
    overview_quantiles_file_list = {}
    overview_quantiles_no_pool_file_list = {}
    overview_quantiles_mean_pool_file_list = {}
    overview_quantiles_max_pool_file_list = {}
    overview_quantiles_q_pool_file_list = {}
    overview_unweighted_max_file_list = {}
    overview_weighted_max_file_list = {}
    overview_spatial_metrics_file_list = {}
    overview_acf_file_list = {}
    overview_psd_file_list = {}
    overview_correlations_file_list = {}
    overview_joint_ex_file_list = {}
    overview_rh_diffs_file_list = {}
    overview_copula_file_list = {}

    for sel_target in variables:
        out_path_tmp = os.path.join(out_path, sel_target)
        os.makedirs(out_path_tmp, exist_ok=True)
        os.makedirs(os.path.join(out_path_tmp, 'txt'), exist_ok=True)

        log_file = os.path.join(out_path_tmp, f"log.txt")

        #file_log(fname=log_file, log_line='Specified args: ' + ''.join(f'\n{k}:\t{v}' for k, v in vars(args).items()))

        out_path_list[sel_target] = out_path_tmp
        log_file_list[sel_target] = log_file

        # 2
        overview_mse_file = os.path.join(out_path_tmp, 'txt', f"2_{sel_target}_overview_MSE_{content_loss}.txt")
        # 3
        overview_es_file = os.path.join(out_path_tmp, 'txt', f"3-1_{sel_target}_overview_ES_{content_loss}.txt")
        overview_es_avg_file = os.path.join(out_path_tmp, 'txt', f"3-2_{sel_target}_overview_ES-avg_{content_loss}.txt")
        # 7
        overview_crps_no_pool_file = os.path.join(out_path_tmp, 'txt', f"7-1_{sel_target}_overview_CRPS-no-pool_{content_loss}.txt")
        overview_crps_max_pool_file = os.path.join(out_path_tmp, 'txt', f"7-2_{sel_target}_overview_CRPS-max-pool_{content_loss}.txt")
        overview_crps_window_pool_file = os.path.join(out_path_tmp, 'txt', f"7-3_{sel_target}_overview_CRPS-window-pool_{content_loss}.txt")
        # 8
        overview_ranks_file = os.path.join(out_path_tmp, 'txt', f"8_{sel_target}_overview_ranks_{content_loss}.txt")
        # 9
        overview_quantile_loss_file = os.path.join(out_path_tmp, 'txt', f"9_{sel_target}_overview_quantile-loss_{content_loss}.txt")
        # 10
        overview_quantiles_file = os.path.join(out_path_tmp, 'txt', f"10-1_{sel_target}_overview_quantiles_{content_loss}.txt")
        overview_quantiles_no_pool_file = os.path.join(out_path_tmp, 'txt', f"10-2_{sel_target}_overview_quantiles-no-pool_{content_loss}.txt")
        overview_quantiles_mean_pool_file = os.path.join(out_path_tmp, 'txt', f"10-3_{sel_target}_overview_quantiles-mean-pool_{content_loss}.txt")
        overview_quantiles_max_pool_file = os.path.join(out_path_tmp, 'txt', f"10-4_{sel_target}_overview_quantiles-max-pool_{content_loss}.txt")
        overview_quantiles_q_pool_file = os.path.join(out_path_tmp, 'txt', f"10-5_{sel_target}_overview_quantiles-q-pool_{content_loss}.txt")
        overview_unweighted_max_file = os.path.join(out_path_tmp, 'txt', f"10-5_{sel_target}_overview_unweighted-max_{content_loss}.txt")
        overview_weighted_max_file = os.path.join(out_path_tmp, 'txt', f"10-6_{sel_target}_overview_weighted-max_{content_loss}.txt")
        overview_spatial_metrics_file = os.path.join(out_path_tmp, 'txt', f"10-7_{sel_target}_overview_spatial_metrics_{content_loss}.txt")
        # 11
        overview_acf_file = os.path.join(out_path_tmp, 'txt', f"11_{sel_target}_overview_acf_{content_loss}.txt")

        # 12
        overview_psd_file = os.path.join(out_path_tmp, 'txt', f"12_{sel_target}_overview_psd_{content_loss}.txt")
        
        overview_mse_file_list[sel_target] = overview_mse_file
        overview_es_file_list[sel_target] = overview_es_file
        overview_es_avg_file_list[sel_target] = overview_es_avg_file
        overview_crps_no_pool_file_list[sel_target] = overview_crps_no_pool_file
        overview_crps_max_pool_file_list[sel_target] = overview_crps_max_pool_file
        overview_crps_window_pool_file_list[sel_target] = overview_crps_window_pool_file
        overview_ranks_file_list[sel_target] = overview_ranks_file
        overview_quantile_loss_file_list[sel_target] = overview_quantile_loss_file
        overview_quantiles_file_list[sel_target] = overview_quantiles_file
        overview_quantiles_no_pool_file_list[sel_target] = overview_quantiles_no_pool_file
        overview_quantiles_mean_pool_file_list[sel_target] = overview_quantiles_mean_pool_file
        overview_quantiles_max_pool_file_list[sel_target] = overview_quantiles_max_pool_file
        overview_quantiles_q_pool_file_list[sel_target] = overview_quantiles_q_pool_file
        overview_unweighted_max_file_list[sel_target] = overview_unweighted_max_file
        overview_weighted_max_file_list[sel_target] = overview_weighted_max_file
        overview_spatial_metrics_file_list[sel_target] = overview_spatial_metrics_file
        overview_acf_file_list[sel_target] = overview_acf_file
        overview_psd_file_list[sel_target] = overview_psd_file
        
        # 14
        overview_joint_ex_file_list[sel_target] = os.path.join(out_path_tmp, 'txt', f"14_{sel_target}_overview_exeedances.txt")
        # 15
        overview_rh_diffs_file_list[sel_target] = os.path.join(out_path_tmp, 'txt', f"15_{sel_target}_overview_rh_diffs.txt")
        # 16
        overview_copula_file_list[sel_target] = os.path.join(out_path_tmp, 'txt', f"16_{sel_target}_overview_copula_vars.txt")


    variable_pairs = [('tas', 'pr'), ('pr', 'sfcWind'), ('sfcWind', 'rsds')] #, ('rsds', 'tas')]
    for sel_targets in variable_pairs:
        os.makedirs(os.path.join(out_path, 'txt'), exist_ok=True)
        overview_correlations_file_list[sel_targets] = os.path.join(out_path, 'txt', f"13_{sel_targets[0]}_{sel_targets[1]}_overview_correlations.txt")
        overview_joint_ex_file_list[sel_targets] = os.path.join(out_path, 'txt', f"14_{sel_targets[0]}_{sel_targets[1]}_overview_joint_ex.txt")
        overview_rh_diffs_file_list[sel_targets] = os.path.join(out_path, 'txt', f"15_{sel_targets[0]}_{sel_targets[1]}_overview_rh_diffs.txt")
    

    ###################################################################################################################
    # GET RCM-GCM COMBINATIONS AND HELPER FOR RUN INDEX

    import concurrent.futures

    def process_gcm_rcm(selected_gcm, selected_rcm):
        variables = args.variables
        variable_pairs = [('tas', 'pr'), ('pr', 'sfcWind'), ('sfcWind', 'rsds')]
        
        # store logs per process (or per task id)
        logs = []
        def file_log(fname, log_line, print_line=True):
            logs.append((fname, log_line, print_line))
        ###################################################################################################################
        # INITIALIZE SUMMARY LISTS FOR ALL RUNS AND VARIABLES
        df_lists_mse = {}
        df_lists_es = {}
        df_lists_crps = {}
        df_lists_rh = {}
        df_lists_spatial_metrics = {}
        df_lists_acfs = {}
        df_lists_pw_corr = {}
        df_lists_rh_diffs = {}
        df_lists_psd = {}
        df_lists_quantiles = {}
        

        ###################################################################################################################
        # Load true data

        if args.project_specifier == 'speed2zero' or args.project_specifier == 'maybritt':
            if args.period == 'train-period':
                data_dir = os.path.join(data_path, 'train')
                mode = "train"
            elif args.period == '2030-2039':
                data_dir = os.path.join(data_path, 'test/interpolation')
                mode = "test_interpolation"
            elif args.period == '2090-2099':
                data_dir = os.path.join(data_path, 'test/extrapolation')
                mode = "test_extrapolation"

        # get time vals
        time_vals = get_dates(selected_rcm, selected_gcm, mode=mode, root = data_path)
        time_vals_months = pd.to_datetime(time_vals).month
        
        load_gan = not args.not_load_gan
        load_diffusion = not args.not_load_diffusion
        true_data, gen_data = load_data(selected_gcm, selected_rcm, variables, mode=mode, run_ids=args.run_id, load_gan=load_gan,
                                        load_benchmarks=not args.not_load_benchmarks, load_diffusion=load_diffusion, 
                                        filter_outliers=args.filter_outliers)
        if not args.not_load_benchmarks:
            methods_stoch = ['analogues', 'idr'] + args.run_id
        else:
            methods_stoch = args.run_id.copy()
        if load_gan:
            methods_stoch += ["gan"]
        if load_diffusion:
            methods_stoch += ["corrdiff"]
            methods_stoch += ["corrdiff_fixed-seeds"]
        methods_all = list(gen_data.keys())
        
        print("loaded following methods: " , methods_all)
        # print(methods_stoch)
        
        if not args.not_load_benchmarks:
            methods_all_with_shuffled = list(gen_data.keys()) + [run + "_shuffled" for run in args.run_id] + ["analogues_shuffled"]
        else:
            methods_all_with_shuffled = list(gen_data.keys()) + [run + "_shuffled" for run in args.run_id]
                    
        if not args.not_load_benchmarks:
            gen_data = add_shuffled_benchmarks(gen_data, methods = args.run_id + ["analogues"], variables = variables)
        else:
            gen_data = add_shuffled_benchmarks(gen_data, methods = args.run_id, variables = variables)

        ###################################################################################################################
        # 1. Print examples
        if args.plot_pred or args.run_all:
            var = "pr"
            scores = cond_energy_score(torch.tensor(true_data[var].copy()).unsqueeze(1), 
                torch.tensor(gen_data[args.run_id[0]][var][:,:,:,0].copy()).unsqueeze(1), 
                torch.tensor(gen_data[args.run_id[0]][var][:,:,:,1].copy()).unsqueeze(1))[0]

            order_scores = np.argsort(scores)
            indices = np.arange(1, 11, 1) * len(order_scores) // 11 - 1

            selected_data_indices =  order_scores[indices] # now pick images according to scores
            selected_samples_indices = np.arange(3)

            for var in variables:
                methods = methods_all.copy()
                #if var not in ['tas', 'pr'] and not args.not_load_gan:
                #    methods.remove("gan")
                #    num_rows = (len(methods_stoch) - 1) * 3 + 1 + len(methods) - (len(methods_stoch) - 1)
                #else:
                num_rows = len(methods_stoch) * 3 + 1 + len(methods) - len(methods_stoch)
                num_cols = 10
                    
                global_vmin = min([np.nanmin(true_data[var][idx]) for idx in selected_data_indices])
                global_vmax = max([np.nanmax(true_data[var][idx]) for idx in selected_data_indices])
                
                if var == "pr":
                    global_vmin = 0
                    global_vmax = global_vmax * 0.95

                fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
                cmap = 'Spectral_r'

                # for j, method in enumerate(["LR", "true"] + methods):
                for i, idx in enumerate(selected_data_indices):
                    row_count = 0
                    for j, method in enumerate(["true"] + methods):
                        # Plot true_data for this index
                        if j == 0:
                            axs[j, i].imshow(true_data[var][idx], cmap=cmap, vmin=global_vmin, vmax=global_vmax)
                            axs[j, i].set_title(f"g.t. high-res. {idx}")
                            row_count += 1
                        elif method in methods_stoch:
                            for s in selected_samples_indices:
                                axs[row_count, i].imshow(gen_data[method][var][idx, :, :, s], cmap=cmap, vmin=global_vmin,
                                                        vmax=global_vmax)
                                axs[row_count, i].set_title(f"{method} {idx} sample {s}")
                                axs[row_count, 0].set_ylabel(f"{method}")
                                row_count += 1
                        else:
                            axs[row_count, i].imshow(gen_data[method][var][idx, :, :, 0], cmap=cmap, vmin=global_vmin,
                                                vmax=global_vmax)
                            axs[row_count, i].set_title(f"{method} {idx}")
                            axs[row_count, 0].set_ylabel(f"{method}")
                            row_count += 1
                                    

                # for j, l in enumerate(["Expected high-res."]+ ["Generated"]*args.num_samples):
                    #   axs[row_count, 0].set_ylabel(l)

                # Add colorbar
                # fig.colorbar(ax_LR, ax=axs[0,-1])
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

                plt.tight_layout()

                if args.project_specifier == 'speed2zero':
                    if args.random_input:
                        plt.savefig(os.path.join(out_path_list[var],
                                                    f"1_{var}_{run_id_name}_{content_loss}_{selected_gcm}_{selected_rcm}_{args.period}_comparison_random_input.png"))
                    else:
                        plt.savefig(os.path.join(out_path_list[var],
                                                    f"1_{var}_{run_id_name}_{content_loss}_{selected_gcm}_{selected_rcm}_{args.period}_pred-id-{args.pred_id}_comparison.png"))

                elif args.project_specifier == 'deepdown':
                    if args.random_input:
                        plt.savefig(os.path.join(out_path_list[var],
                                                    f"1_{var}_{run_id_name}_{content_loss}_{args.period}_comparison_random_input.png"))
                    else:
                        plt.savefig(os.path.join(out_path_list[var],
                                                    f"1_{var}_{run_id_name}_{content_loss}_{args.period}_pred-id-{args.pred_id}_comparison.png"),

                                    dpi=300)
                
                elif args.project_specifier == 'maybritt':
                    plt.savefig(os.path.join(out_path_list[var],
                                                f"1_{var}_{selected_gcm}_{selected_rcm}_{run_id_name}_{args.period}_comparison_sorted.png"))

        ##################################################################################################################            
        # 2. MSE
        if args.calc_mse or args.run_all:
            for var in variables:
                methods = methods_all.copy()
                #if var not in ['tas', 'pr'] and not args.not_load_gan:
                #    methods.remove("gan")
                    
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 2. MSE for {var} {selected_gcm}_{selected_rcm}_{args.period}")
                # res_mse = eval_mse(true_data, gen_data, methods=methods, var=var, n_samples=args.num_samples)
                # eval_func, true_data, gen_data, var="tas", methods=None, n_bootstrap=100, sample_frac=1.0, **eval_kwargs
                res_mse = bootstrap_eval_eff(eval_mse_boot, true_data, gen_data, methods=methods, var=var, n_samples=args.num_samples,
                                            n_bootstrap=100, sample_frac=1)
                file_log(fname=log_file_list[var], log_line=res_mse)

                if args.project_specifier == 'speed2zero':
                        output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_mse}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_mse}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{res_mse}'

                file_log(fname=overview_mse_file_list[var],
                            log_line=output_line,
                            print_line=False)

                
                
                res_mse["rcm"] = selected_rcm
                res_mse["gcm"] = selected_gcm
                df_lists_mse[var] = res_mse

        ##################################################################################################################
        # 3. Energy score
        if args.calc_es or args.run_all:
            for var in variables:
                methods = methods_all.copy()
                # if var not in ['tas', 'pr']  and not args.not_load_gan:
                #    methods.remove("gan")
                    
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 3. Energy score for {var} {selected_gcm}_{selected_rcm}_{args.period}")
                # es = eval_es(true_data, gen_data, var=var, methods=methods)
                es = bootstrap_eval_eff(eval_es_boot, true_data, gen_data, methods=methods, var=var,
                                            n_bootstrap=100, sample_frac=1)
                file_log(fname=log_file_list[var], log_line=f"{es = }\n")
                
                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{es}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{es}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{es}'

                file_log(fname=overview_es_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
                es["rcm"] = selected_rcm
                es["gcm"] = selected_gcm
                df_lists_es[var] = es

        ##################################################################################################################
        # 7. CRPS
        if args.calc_crps or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 7. CRPS for {var} {selected_gcm}_{selected_rcm}_{args.period}")
                
                methods = methods_all.copy()
                # if var not in ['tas', 'pr']  and not args.not_load_gan:
                #    methods.remove("gan")
                # Max pooling 10x10
                # crps_max_pool_10 = eval_crps(true_data, gen_data, var=var, methods=methods,
                #                          pool_method="max", kernel_size=10)
                crps_max_pool_10 = bootstrap_eval_eff(eval_crps_boot, true_data, gen_data, var=var, methods=methods,
                                            pool_method="max", kernel_size=10, n_bootstrap=100, sample_frac=1)

                # CRPS pixelwise, no pooling
                # crps_no_pool = eval_crps(true_data, gen_data, var=var, methods=methods,
                #                         pool_method=None)
                crps_no_pool = bootstrap_eval_eff(eval_crps_boot, true_data, gen_data, var=var, methods=methods,
                                            pool_method=None, n_bootstrap=100, sample_frac=1)

                crps_results = {}
                crps_results[var] = pd.concat([crps_no_pool, crps_max_pool_10], axis=1)
                file_log(fname=log_file_list[var], log_line=f"{var = }")
                file_log(fname=log_file_list[var], log_line=f"CRPS results:\n{crps_results[var]}")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{crps_results[var].to_string()}'

                file_log(fname=overview_crps_window_pool_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
                crps_results[var]["rcm"] = selected_rcm
                crps_results[var]["gcm"] = selected_gcm
                df_lists_crps[var] = crps_results[var]

        ##################################################################################################################
        # 8. Rank histogram
        if args.plot_ranks or args.run_all:
            for var in variables:
                methods = methods_stoch.copy()

                # spatial mean
                ground_truth = true_data[var].mean(axis = (1, 2))
                forecasts = {method: gen_data[method][var][..., :args.num_samples].mean(axis = (1, 2)) for method in methods}
                # forecasts_mean = {method: gen_data[method][var][..., :args.num_samples].mean(axis = (1, 2)) for method in methods}
                rh_sp_mean_summary, hists_sp_mean = eval_rank_histogram_agg(ground_truth, forecasts, methods=methods, full=True, prefix="sp_mean_")
                for method in methods:
                    np.save(os.path.join(out_path_list[var], f"8_{var}_{selected_gcm}_{selected_rcm}_{args.period}_{method}_rh_sp_mean.npy"), hists_sp_mean[method])
                
                # spatial max
                ground_truth = true_data[var].max(axis = (1, 2))
                forecasts = {method: gen_data[method][var][..., :args.num_samples].max(axis = (1, 2)) for method in methods}
                rh_sp_max_summary, hists_sp_max = eval_rank_histogram_agg(ground_truth, forecasts, methods=methods, full=True, prefix="sp_max_")
                for method in methods:
                    np.save(os.path.join(out_path_list[var], f"8_{var}_{selected_gcm}_{selected_rcm}_{args.period}_{method}_rh_sp_max.npy"), hists_sp_max[method])
                    
                # MCB for locations
                mcb_locs, means_locs, var_locs, mcb_last_locs, mcb_first_locs, mcb_last_locs_signed, mcb_first_locs_signed \
                    = eval_rh_perloc(true_data, gen_data, var = var, methods = methods, num_locs=128*128, num_samples=9)
                mcb_df = pd.DataFrame({"loc_avg_mcb": [mcb_locs[method].mean() for method in methods],
                                        "loc_avg_mcb_last_bin": [mcb_last_locs[method].mean() for method in methods],
                                        "loc_avg_mcb_first_bin": [mcb_first_locs[method].mean() for method in methods],
                                        "loc_avg_mcb_last_bin_signed": [mcb_last_locs_signed[method].mean() for method in methods],
                                        "loc_avg_mcb_first_bin_signed": [mcb_first_locs_signed[method].mean() for method in methods],
                                        }, index = methods)
                
                rk_summary = pd.concat([rh_sp_mean_summary, rh_sp_max_summary, mcb_df], axis = 1)
                file_log(fname=log_file_list[var], log_line=f"{rk_summary = }\n")
                                            
                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{rk_summary}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{rk_summary}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{rk_summary}'

                file_log(fname=overview_ranks_file_list[var], log_line=output_line, print_line=False)
                
                rk_summary["rcm"] = selected_rcm
                rk_summary["gcm"] = selected_gcm
                df_lists_rh[var] = rk_summary

        #######################################################################################################
        # 10. Spatial metrics
        
        if args.calc_spatial_metrics or args.run_all:
            for var in variables:
                methods = methods_all.copy()
                #if var not in ['tas', 'pr'] and not args.not_load_gan:
                #    methods.remove("gan")
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 10. Spatial metrics for {var} {selected_gcm}_{selected_rcm}_{args.period}")

                # ql_no_pooling = eval_quantile_loss(true_data, gen_data, var=var, methods=methods, pool_method=None)
                ql_no_pooling = bootstrap_eval_eff(eval_quantile_loss_boot, true_data, gen_data, var=var, methods=methods, pool_method=None,
                                                    n_bootstrap=100, sample_frac=1, quantiles = [0.1, 0.5, 0.9])
                file_log(fname=log_file_list[var], log_line=f"{ql_no_pooling = }\n")
                file_log(fname=overview_quantiles_no_pool_file_list[var],
                            log_line=f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_no_pooling}',
                            print_line=False)

                #ql_mean_pooling = eval_quantile_loss(true_data, gen_data, var=var, methods = methods, pool_method="mean")
                ql_mean_pooling = bootstrap_eval_eff(eval_quantile_loss_boot, true_data, gen_data, var=var, methods=methods, pool_method="mean",
                                                        n_bootstrap=100, sample_frac=1, quantiles =  [0.1, 0.5, 0.9])
                
                file_log(fname=log_file_list[var], log_line=f"{ql_mean_pooling = }\n")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_mean_pooling}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_mean_pooling}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{ql_mean_pooling}'

                file_log(fname=overview_quantiles_mean_pool_file_list[var], log_line=output_line, print_line=False)
                
                # max pooling
                ql_max_pooling = bootstrap_eval_eff(eval_quantile_loss_boot, true_data, gen_data, var=var, methods=methods, pool_method="max",
                                                        n_bootstrap=100, sample_frac=1, quantiles = [0.1, 0.5, 0.9])
                
                file_log(fname=log_file_list[var], log_line=f"{ql_max_pooling = }\n")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_max_pooling}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_max_pooling}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{ql_max_pooling}'

                file_log(fname=overview_quantiles_max_pool_file_list[var], log_line=output_line, print_line=False)
                
                # q pool
                # overview_quantiles_q_pool_file_list
                ql_q_pooling = bootstrap_eval_eff(eval_quantile_loss_boot, true_data, gen_data, var=var, methods=methods, pool_method="quantile",
                                                    pool_quantile = 0.9, n_bootstrap=100, sample_frac=1, quantiles =  [0.1, 0.5, 0.9])
                
                file_log(fname=log_file_list[var], log_line=f"{ql_q_pooling = }\n")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_q_pooling}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{ql_q_pooling}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{ql_q_pooling}'

                file_log(fname=overview_quantiles_q_pool_file_list[var], log_line=output_line, print_line=False)

                res_spatial_metrics = pd.concat([ql_no_pooling, ql_mean_pooling, ql_max_pooling, ql_q_pooling], axis = 1)
                res_spatial_metrics["rcm"] = selected_rcm
                res_spatial_metrics["gcm"] = selected_gcm
                df_lists_spatial_metrics[var] = res_spatial_metrics

        #######################################################################################################
        # 11. ACF

        if args.calc_acf or args.run_all:
            for var in variables:
                methods = methods_all.copy()
                #if var not in ['tas', 'pr'] and not args.not_load_gan:
                #    methods.remove("gan")
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 11. Temporal autocorrelation for {var} {selected_gcm}_{selected_rcm}_{args.period}")

                # Flexible conditional quantile loss function
                acf_no_pool = eval_acf(true_data, gen_data, var=var, methods=methods, pool_method=None, kernel_size=1)
                acf_mean_pool = eval_acf(true_data, gen_data, var=var, methods=methods, pool_method="average", kernel_size=10)
                acf_sp_mean = eval_acf(true_data, gen_data, var=var, methods=methods, pool_method="average", kernel_size=128)
                
                res_acf = pd.concat([acf_no_pool, acf_mean_pool, acf_sp_mean], axis = 1)
                
                file_log(fname=log_file_list[var], log_line=f"{res_acf = }\n")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_acf}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_acf}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{res_acf}'

                file_log(fname=overview_acf_file_list[var], log_line=output_line, print_line=False)

                
                res_acf["rcm"] = selected_rcm
                res_acf["gcm"] = selected_gcm
                df_lists_acfs[var] = res_acf

        
        ##############################################################################################################

        # 12. Power spectra
        if args.plot_power_spectrum or args.run_all:
            methods = methods_all.copy()
            fig, axs = plt.subplots(len(methods), len(variables), figsize=(10, 10))
            axs = np.reshape(axs, (len(methods), len(variables)))

            # if len(variables) == 1:
            #     axs = np.reshape(axs, (1, len(methods)))

            for i, var in enumerate(variables):
                
                file_log(fname=log_file_list[var],
                    log_line=f"\n---------- 12. Power spectra for {var} {selected_gcm}_{selected_rcm}_{args.period}")

                # old version - 
                #res_psd = bootstrap_eval_eff(eval_psd_boot, true_data, gen_data, methods=methods, 
                #                             var=var, n_samples=1,
                #                            n_bootstrap=100, sample_frac=1)
                
                res_psd, rapsd_true, rapsd_gen_dict = bootstrap_eval_eff_psd(true_data, gen_data, var=var, methods=methods, n_samples=1)
                file_log(fname=log_file_list[var], log_line=res_psd)
                                    
                # also save PSDs and plot
                # old version - had to rerun
                #results, rapsd_true, rapsd_gen_dict = eval_psd_boot(true_data, gen_data, var=var, methods=methods, n_samples=1, verbose=True)
                
                true_psds_avg = rapsd_true.mean(axis=0)
                np.save(os.path.join(out_path_list[var], 
                                        f"12_{var}_{selected_gcm}_{selected_rcm}_{args.period}_true_psd.npy"), 
                        true_psds_avg)
                
                for j, method in enumerate(methods):
                    rapsd_gen_avg = rapsd_gen_dict[method].mean(axis=0)
                    np.save(os.path.join(out_path_list[var], 
                                            f"12_{var}_{selected_gcm}_{selected_rcm}_{args.period}_{method}_gen_psd.npy"), 
                            rapsd_gen_avg)
                    
                    im = axs[j, i].plot(true_psds_avg, label="True")
                    im = axs[j, i].plot(rapsd_gen_avg, label="Generated")
                    axs[j, i].set_title(f'RAPS for {method} and {var}')
                    axs[j, i].set_xscale('log')
                    axs[j, i].set_yscale('log')
                    axs[j, i].legend()

                # logging and saving for each variable
                file_log(fname=log_file_list[var], log_line=f"{res_psd = }\n")
            
                if args.project_specifier == 'speed2zero':
                        output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_psd}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{res_psd}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{res_psd}'
                
                res_psd["rcm"] = selected_rcm
                res_psd["gcm"] = selected_gcm
                
                file_log(fname=overview_psd_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
                df_lists_psd[var] = res_psd
                    
            # save joint plot
            plt.tight_layout()
            
            if len(variables) > 1:
                fname_prefix = 'joint-target'
            else:
                fname_prefix = variables[0]

            if args.project_specifier == 'speed2zero':
                output_fname = f"12_{fname_prefix}_{run_id_name}_{content_loss}_{selected_gcm}_{selected_rcm}_{args.period}_pred-id-{args.pred_id}_power_spectra.png"
            elif args.project_specifier == 'deepdown':
                output_fname = f"12_{fname_prefix}_{run_id_name}_{content_loss}_{args.period}_pred-id-{args.pred_id}_power_spectra.png"
            elif args.project_specifier == 'maybritt':
                output_fname = f"12_{fname_prefix}_{run_id_name}_{selected_gcm}_{selected_rcm}_{args.period}_power_spectra.png"

            plt.savefig(os.path.join(out_path, output_fname))
            

        ##################################################################################################################
        # 17. Quantiles
        if args.calc_quantiles or args.run_all:
            for var in variables:
                methods = methods_all.copy()
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 17. Quantiles for {var} {selected_gcm}_{selected_rcm}_{args.period}")

                quantiles_winter = eval_quantiles_marginal(true_data, gen_data, var=var, methods=methods, 
                                                    quantiles=[0.05, 0.95], months = [12,1,2], time_vals_months = time_vals_months)
                
                quantiles_summer = eval_quantiles_marginal(true_data, gen_data, var=var, methods=methods, 
                                                    quantiles=[0.05, 0.95], months = [6,7,8], time_vals_months = time_vals_months)
                
                quantiles = pd.concat([quantiles_winter, quantiles_summer], axis = 1)
                file_log(fname=log_file_list[var], log_line=f"{quantiles = }\n")

                if args.project_specifier == 'speed2zero':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{quantiles}'
                elif args.project_specifier == 'deepdown':
                    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{quantiles}'
                elif args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{quantiles}'

                file_log(fname=overview_quantiles_file_list[var], log_line=output_line, print_line=False)
                
                quantiles["rcm"] = selected_rcm
                quantiles["gcm"] = selected_gcm
                df_lists_quantiles[var] = quantiles
        
        ##################################################################################################################
        # 13. Correlations between variables
        
        if args.calc_corrs or args.run_all:
            for var_pair in variable_pairs:
                var = var_pair[0]
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 13. correlation for {var_pair[0]} with {var_pair[1]} {selected_gcm}_{selected_rcm}_{args.period}")


                methods = methods_all_with_shuffled.copy()
                # methods.remove("gan")
                pw_corr_no_pool = eval_pointwise_corr(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1], methods = methods,
                                    pool_method=None, kernel_size=1, j=0)
                
                pw_corr_max_pool_10 = eval_pointwise_corr(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1], methods = methods,
                                    pool_method='max', kernel_size=10, j=0)

                pw_corr_avg_pool_10 = eval_pointwise_corr(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1], methods = methods,
                                    pool_method='average', kernel_size=10, j=0)
                
                pw_corr_results = {}
                pw_corr_results[var_pair] = pd.DataFrame({"PW corrs pixelwise": pw_corr_no_pool.iloc[:,0],
                                                "PW corrs max-pool 10x10": pw_corr_max_pool_10.iloc[:,0],
                                                "PW corrs avg-pool 10x10": pw_corr_avg_pool_10.iloc[:,0]},
                                                index=methods)
                    
                file_log(fname=log_file_list[var], log_line=f"{var = }")
                file_log(fname=log_file_list[var], log_line=f"PW corrs results with {var_pair[1]}:\n{pw_corr_results[var_pair]}")

                #if args.project_specifier == 'speed2zero':
                #    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                #elif args.project_specifier == 'deepdown':
                #    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                if args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{pw_corr_results[var_pair].to_string()}'

                file_log(fname=overview_correlations_file_list[var_pair],
                            log_line=output_line,
                            print_line=False)
                
                pw_corr_results[var_pair]["rcm"] = selected_rcm
                pw_corr_results[var_pair]["gcm"] = selected_gcm
                df_lists_pw_corr[var_pair] = pw_corr_results[var_pair]
                
        ##################################################################################################################
        # 14. Joint exceedances
        #variable_pairs = [('tas', 'pr'), ('pr', 'sfcWind'), ('sfcWind', 'rsds')]
        #modes = [('ex', 'sub'), ('ex', 'ex'), ('sub', 'sub')]
        #quantiles1 = [(0.8, 0.4), (0.7, 0.7), (0.3, 0.3)]
        # quantiles2 = [(0.99, 0.3), (0.99, 0.99), (0.01, 0.01)]
        if args.calc_joint_ex or args.run_all:
            
            for var in variables:
                """
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 14-1. Marginal-marginal exceedances for {var} {selected_gcm}_{selected_rcm}_{args.period}")
                marginal_ex1 = eval_pw_exceedance_marginal_marginal(true_data, gen_data, var1 = var,
                    methods=methods_all, q1=0.9,
                    month_array=time_vals_months, months=np.arange(1,13), return_individual_months=False)
                
                conditional_ex1 = eval_pw_exceedance_marginal_conditional(true_data, gen_data, var1 = var,
                    methods=methods_all, q1=0.9, mode1= "ex",
                    month_array=time_vals_months, months=np.arange(1,13), return_individual_months=False)
                
                ex_res = pd.concat([marginal_ex1, conditional_ex1], axis=1)
                
                file_log(fname=log_file_list[var], log_line=f"{var = }")
                file_log(fname=log_file_list[var], log_line=f"Exceedances results:\n{ex_res}")
                
                if args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{ex_res.to_string()}'
                
                file_log(fname=overview_joint_ex_file_list[var],
                            log_line=output_line,
                            print_line=False)                    
            
                """
            
            variable_pairs = [('pr', 'sfcWind')]
            modes = [('ex', 'ex')]
            quantiles_list = [[(0.5, 0.5)], [(0.6, 0.6)], [(0.7, 0.7)], [(0.8, 0.8)]]#, [(0.9, 0.9)], [(0.95, 0.95)]]

            for i, var_pair in enumerate(variable_pairs):
                joint_ex_results = {}
                var = var_pair[0]
                
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 14. Joint exceedances for {var_pair[0]} with {var_pair[1]} {selected_gcm}_{selected_rcm}_{args.period}")
                for j, quantiles1 in enumerate(quantiles_list):
                    joint_ex_results[j] = {}

                    joint_ex1 =  eval_pw_exceedance_joint_marginal(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1],
                                                                methods=methods_all_with_shuffled, q1 = quantiles1[i][0], q2 = quantiles1[i][1],
                                                                month_array=time_vals_months, months=np.arange(1, 2),
                                                                return_individual_months=False)

                    joint_ex2 = eval_pw_exceedance_joint_conditional(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1], 
                                                                    methods = methods_all_with_shuffled,
                                                                    q1 = quantiles1[i][0], q2 = quantiles1[i][1], mode1 = modes[i][0], mode2 = modes[i][1],
                                                                    #month_array=time_vals_months, months=np.arange(1,13), return_individual_months=False)
                                                                    month_array=time_vals_months, months=np.arange(1,2), return_individual_months=False)
                    
                    #joint_ex2 = eval_pw_exceedance_joint_conditional(true_data, gen_data, var1 = var_pair[0], var2 = var_pair[1], 
                    #                                            methods = methods_all_with_shuffled,
                    #                                     q1 = quantiles2[i][0], q2 = quantiles2[i][1], mode1 = modes[i][0], mode2 = modes[i][1],
                    #                                     month_array=time_vals_months, months=np.arange(1,13), return_individual_months=False)
                    
                    joint_ex_results[j] = pd.concat([joint_ex1, joint_ex2], axis=1)
                
                joint_ex_results = pd.concat([joint_ex_results[j] for j in range(len(joint_ex_results))], axis=1)
                
                file_log(fname=log_file_list[var], log_line=f"{var = }")
                file_log(fname=log_file_list[var], log_line=f"Joint ex results with {var_pair[1]}:\n{joint_ex_results}")

                #if args.project_specifier == 'speed2zero':
                #    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                #elif args.project_specifier == 'deepdown':
                #    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                if args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{joint_ex_results.to_string()}'

                file_log(fname=overview_joint_ex_file_list[var_pair],
                        log_line=output_line,
                        print_line=False)

                corr_res = eval_conditional_corr(gen_data, var1 = var_pair[0], var2 = var_pair[1], methods = methods_all_with_shuffled,
                        month_array=time_vals_months, months=np.arange(1,13), base_method = "analogues", return_individual_months=True)
                
                file_log(fname=log_file_list[var], log_line=f"Joint ex results with {var_pair[1]}:\n{corr_res}")

                if args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{corr_res.to_string()}'

                file_log(fname=overview_joint_ex_file_list[var_pair],
                        log_line=output_line,
                        print_line=False)
                
        ###################################################################################################################  
        # 15. Differences between variables
        if args.rh_var_diffs or args.run_all:
            for var_pair in variable_pairs:
                var = var_pair[0]
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 15. calibration of difference for {var_pair[0]} minus {var_pair[1]} {selected_gcm}_{selected_rcm}_{args.period}")

                # methods_all_with_shuffled = methods_all.copy() # for now no shuffling added, not needed for -13         
                primitive_standardization = False
                methods = methods_all_with_shuffled.copy()
                
                """
                if primitive_standardization:
                    ground_truth_spatial_mean_var1 = true_data[var_pair[0]].mean(axis=(-2, -1))
                    ground_truth_spatial_mean_var2 = true_data[var_pair[1]].mean(axis=(-2, -1))
                    mean_true_var1 = ground_truth_spatial_mean_var1.mean()
                    std_true_var1 = ground_truth_spatial_mean_var1.std()                    
                    mean_true_var2 = ground_truth_spatial_mean_var2.mean()
                    std_true_var2 = ground_truth_spatial_mean_var2.std()
                    
                    ground_truth_var1 = (ground_truth_spatial_mean_var1 - mean_true_var1) / std_true_var1
                    ground_truth_var2 = (ground_truth_spatial_mean_var2 - mean_true_var2) / std_true_var2                    
                    ground_truth = ground_truth_var1 - ground_truth_var2
                    
                    forecasts_spatial_mean_var1 = {method: gen_data[method][var_pair[0]][..., :args.num_samples].mean(axis=(-3, -2)) for method in methods_all_with_shuffled}
                    forecasts_spatial_mean_var2 = {method: gen_data[method][var_pair[1]][..., :args.num_samples].mean(axis=(-3, -2)) for method in methods_all_with_shuffled}                    
                    mean_gen_var1 = {method: forecasts_spatial_mean_var1[method].mean() for method in methods_all_with_shuffled}
                    std_gen_var1 = {method: forecasts_spatial_mean_var1[method].std() for method in methods_all_with_shuffled}                    
                    mean_gen_var2 = {method: forecasts_spatial_mean_var2[method].mean() for method in methods_all_with_shuffled}
                    std_gen_var2 = {method: forecasts_spatial_mean_var2[method].std() for method in methods_all_with_shuffled}
                    
                    forecasts_var1 = {method: (forecasts_spatial_mean_var1[method] - mean_gen_var1[method]) / std_gen_var1[method] for method in methods_all_with_shuffled}
                    forecasts_var2 = {method: (forecasts_spatial_mean_var2[method] - mean_gen_var2[method]) / std_gen_var2[method] for method in methods_all_with_shuffled}
                    forecasts = {method: forecasts_var1[method] - forecasts_var2[method] for method in methods_all_with_shuffled}
                else:
                    pass
                # ground_truth = np.mean(ground_truth_full, axis=(-2, -1))
                # forecasts = {method: np.mean(forecasts_full[method], axis=(-2, -1)) for method in methods_all_with_shuffled}
                rh_diff_sp_mean, _ = eval_rank_histogram_agg(ground_truth, forecasts, methods = methods_all_with_shuffled)
                """
                mcb_locs, means_locs, var_locs = eval_rh_var_diffs_perloc(true_data, gen_data, var1=var, var2=var_pair[1],
                                    methods=methods, num_locs=1000)
                            
                mcb_mean_over_locs = []
                for method in methods_all_with_shuffled:
                    mcb_mean_over_locs.append(np.nanmean(mcb_locs[method]))
                    
                rh_mean_over_locs = []
                for method in methods_all_with_shuffled:
                    rh_mean_over_locs.append(np.nanmean(means_locs[method]))
                
                rh_var_over_locs = []
                for method in methods_all_with_shuffled:
                    rh_var_over_locs.append(np.nanmean(var_locs[method]))
                                        
                rh_diff_locs = pd.DataFrame({"mcb_loc_average": mcb_mean_over_locs,
                                                "rh_mean_loc_average": rh_mean_over_locs,
                                            "rh_var_loc_average": rh_var_over_locs
                                                }, index = methods_all_with_shuffled)
                
                rh_diff_results = {}
                rh_diff_results[var_pair] = rh_diff_locs 
                                            #pd.concat([rh_diff_sp_mean,
                                            #           rh_diff_locs], axis=1) 
                                #pd.DataFrame({"RH diff of spatial mean": rh_diff_sp_mean.iloc[:,0]},
                                            #    index=methods_all_with_shuffled)
                
                file_log(fname=log_file_list[var], log_line=f"{var = }")
                file_log(fname=log_file_list[var], log_line=f"RH diff of spatial mean with {var_pair[1]}:\n{rh_diff_results[var_pair]}")

                #if args.project_specifier == 'speed2zero':
                #    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                #elif args.project_specifier == 'deepdown':
                #    output_line = f'{args.period}_{run_id_name}_pred-id-{args.pred_id}\n{crps_results[var].to_string()}'
                if args.project_specifier == 'maybritt':
                    output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{rh_diff_results[var_pair].to_string()}'

                file_log(fname=overview_rh_diffs_file_list[var_pair],
                            log_line=output_line,
                            print_line=False)
                
                rh_diff_results[var_pair]["rcm"] = selected_rcm
                rh_diff_results[var_pair]["gcm"] = selected_gcm
                df_lists_rh_diffs[var_pair]  = rh_diff_results[var_pair]
                
                
        ###################################################################################################################  
        # 16. Copula between variables
        if args.calc_es_copula_vars or args.run_all:
            file_log(fname=log_file_list[var],
                        log_line=f"\n---------- 16. Copula of variables {selected_gcm}_{selected_rcm}_{args.period}")

                # methods_all_with_shuffled = methods_all.copy() # for now no shuffling added, not needed for -13         
                
            var1 = "tas"
            copula_locs = {method: np.empty((true_data[var1].shape[-2], true_data[var1].shape[-1])) for method in methods_all_with_shuffled}
            ground_truth_locs = {}
            forecasts_locs = {}
            for var in variables:
                ground_truth_locs[var] = np.empty((true_data[var1].shape[0], true_data[var1].shape[-2], true_data[var1].shape[-1]))
                forecasts_locs[var] = {method: np.empty((true_data[var1].shape[0], true_data[var1].shape[-2], true_data[var1].shape[-1], args.num_samples)) for method in methods_all_with_shuffled}
            
            # Define a random subset of possible indices for i and j
            num_indices = 100
            possible_indices = [(i, j) for i in range(true_data[var1].shape[-2]) for j in range(true_data[var1].shape[-1])]
            selected_indices = np.random.choice(len(possible_indices), num_indices, replace=False)
            selected_indices = [possible_indices[idx] for idx in selected_indices]
            for i in range(true_data[var1].shape[-2]):
                for j in range(true_data[var1].shape[-1]):
                    if (i, j) not in selected_indices:
                        for method in methods_all_with_shuffled:
                            copula_locs[method][i, j] = np.nan
                    
                    else:
                        for var in variables:
                            ground_truth_locs[var][:, i, j] = ecdf_full(true_data[var][..., i, j])
                            for method in methods_all_with_shuffled:
                                forecasts_locs[var][method][:, i, j, :] = ecdf_full(gen_data[method][var][..., i, j, :args.num_samples], flatten = True)
                            
                        # compute ES on joint vector
                        stacked_ground_truth = np.stack([ground_truth_locs[var][:, i,j] for var in variables], axis=-1)
                        stacked_forecasts = {method: np.stack([forecasts_locs[var][method][:, i,j] for var in variables], axis=1) for method in methods_all_with_shuffled}
                        
                        for method in methods_all_with_shuffled:
                            es_copula = cond_energy_score_batch(torch.tensor(stacked_ground_truth), 
                                                                torch.tensor(stacked_forecasts[method][..., 0]),
                                                                torch.tensor(stacked_forecasts[method][..., 1]))[0]
                            copula_locs[method][i, j] = es_copula #rh_diff_per_loc.loc[method, "mcb"]

            es_mean_over_locs = []
            for method in methods_all_with_shuffled:
                es_mean_over_locs.append(np.nanmean(copula_locs[method]))
                
            # add ES spatial mean
            #stacked_ground_truth_sp = np.concatenate([ground_truth_locs[var] for var in variables], axis=2)
            #stacked_forecasts = {method: np.concatenate([forecasts_locs[var][method] for var in variables], axis=2) for method in methods_all_with_shuffled}
            #es_copula_sp_mean = [cond_energy_score_batch(torch.tensor(), torch.tensor(stacked_forecasts[method]))[0] for method in methods_all_with_shuffled]
            es_copula_sp_mean = [0 for method in methods_all_with_shuffled]
            
            es_copula_res = pd.DataFrame({"es_loc_average": es_mean_over_locs,
                                            "es_sp_mean": es_copula_sp_mean}, index = methods_all_with_shuffled)
            
            
            file_log(fname=log_file_list[var], log_line=f"{var = }")
            file_log(fname=log_file_list[var], log_line=f"ES on copula of variables:\n{es_copula_res}")

            if args.project_specifier == 'maybritt':
                output_line = f'{selected_gcm}_{selected_rcm}_{args.period}_{run_id_name}\n{es_copula_res.to_string()}'

            file_log(fname=overview_copula_file_list[var1],
                        log_line=output_line,
                        print_line=False)
            
            es_copula_res["rcm"] = selected_rcm
            es_copula_res["gcm"] = selected_gcm
            df_lists_copula_vars = es_copula_res
                
        # Collect results for computed metrics
        results_dict = {}
        if args.calc_mse or args.run_all:
            results_dict['mse'] = df_lists_mse
        if args.calc_es or args.run_all:
            results_dict['es'] = df_lists_es
        if args.calc_crps or args.run_all:
            results_dict['crps'] = df_lists_crps
        if args.plot_ranks or args.run_all:
            results_dict['rh'] = df_lists_rh
        if args.calc_spatial_metrics or args.run_all:
            results_dict['spatial_metrics'] = df_lists_spatial_metrics
        if args.calc_acf or args.run_all:
            results_dict['acfs'] = df_lists_acfs
        if args.plot_power_spectrum or args.run_all:
            results_dict['psd'] = df_lists_psd
        if args.calc_quantiles or args.run_all:
            results_dict['quantiles'] = df_lists_quantiles
        if args.calc_corrs or args.run_all:
            results_dict['pw_corr'] = df_lists_pw_corr
        if args.rh_var_diffs or args.run_all:
            results_dict['rh_diffs'] = df_lists_rh_diffs
        if args.calc_es_copula_vars or args.run_all:
            results_dict['copula_vars'] = df_lists_copula_vars
        return results_dict, logs 

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for selected_gcm in gcm_rcm_dict.keys():
            for selected_rcm in gcm_rcm_dict[selected_gcm]:
                if args.gcm == 'all' and args.rcm == 'all':
                    pass
                elif not (selected_gcm == args.gcm and selected_rcm == args.rcm):
                    continue
                futures.append(executor.submit(process_gcm_rcm, selected_gcm, selected_rcm))
        # Optionally collect results

        all_results = []
        all_logs = []

        for future in futures:
            results_dict, logs = future.result()
            all_results.append(results_dict)
            all_logs.append(logs)

        # Concatenate results for each metric and variable
        # Initialize dicts to hold concatenated DataFrames
        concatenated_results = defaultdict(dict)

        for key in all_results[0].keys():
            for var in all_results[0][key].keys():
                dfs = [res[key][var] for res in all_results if key in res and var in res[key]]
                concatenated_results[key][var] = dfs
        # Now you can use concatenated_results["es"]["tas"], concatenated_results["mse"]["pr"], etc.
        # For example, to replace df_lists_es, df_lists_mse, etc.:
        df_lists_es = concatenated_results.get("es", {})
        df_lists_mse = concatenated_results.get("mse", {})
        df_lists_crps = concatenated_results.get("crps", {})
        df_lists_rh = concatenated_results.get("rh", {})
        df_lists_spatial_metrics = concatenated_results.get("spatial_metrics", {})
        df_lists_acfs = concatenated_results.get("acfs", {})
        df_lists_psd = concatenated_results.get("psd", {})
        df_lists_quantiles = concatenated_results.get("quantiles", {})
        df_lists_pw_corr = concatenated_results.get("pw_corr", {})
        df_lists_rh_diffs = concatenated_results.get("rh_diffs", {})
        df_lists_copula_vars = concatenated_results.get("copula_vars", {})
        
    
        for logs in all_logs:   # keeps executor order
            for fname, log_line, print_line in logs:
                if print_line:
                    print(log_line)
                with open(fname, "a") as f:
                    print(log_line, file=f)

                    
        ###### GET AVERAGES OVER ALL RUNS FOR SOME METRICS #####
        
        ##################################################################################################################
        # 2. MSE
        if args.calc_mse or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 2. MSE for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_mse[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_mse_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 3. ES
        if args.calc_es or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 3. ES for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_es[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_es_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 7. CRPS
        if args.calc_crps or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 7. CRPS for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_crps[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_crps_window_pool_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 8. RH summary
        if args.plot_ranks or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 8. Rank hisograms for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_rh[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_ranks_file_list[var],
                            log_line=output_line,
                            print_line=False)        
        
        ##################################################################################################################
        # 10. Spatial metrics
        if args.calc_spatial_metrics or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 10. Spatial metrics for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_spatial_metrics[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_spatial_metrics_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        
        ##################################################################################################################
        # 11. ACF
        if args.calc_acf or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 11. ACFs {var} averaged {args.period}")
                df_all = postproc_df(df_lists_acfs[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_acf_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 12. PSD
        if args.plot_power_spectrum or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 12. PSD for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_psd[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_psd_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 17. Quantiles
        if args.calc_quantiles or args.run_all:
            for var in variables:
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 17. Quantiles for {var} averaged {args.period}")
                df_all = postproc_df(df_lists_quantiles[var])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_quantiles_file_list[var],
                            log_line=output_line,
                            print_line=False)
                
        ##################################################################################################################
        # 13. Correlations between variables
        
        if args.calc_corrs or args.run_all:
            for var_pair in variable_pairs:
                var = var_pair[0]
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 13. correlation for {var_pair[0]} with {var_pair[1]} averaged {args.period}")
                df_all = postproc_df(df_lists_pw_corr[var_pair])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_correlations_file_list[var_pair],
                            log_line=output_line,
                            print_line=False)
        
        ##################################################################################################################
        # 15. RH differences
        
        if args.rh_var_diffs or args.run_all:
            for var_pair in variable_pairs:
                var = var_pair[0]
                file_log(fname=log_file_list[var],
                            log_line=f"\n---------- 15. calibration of difference for {var_pair[0]} with {var_pair[1]} averaged {args.period}")
                df_all = postproc_df(df_lists_rh_diffs[var_pair])
                file_log(fname=log_file_list[var], log_line=df_all)

                if args.project_specifier == 'maybritt':
                    output_line = f'average_{args.period}_{run_id_name}\n{df_all.to_string()}'

                file_log(fname=overview_rh_diffs_file_list[var_pair],
                            log_line=output_line,
                            print_line=False)
        
        
        ##################################################################################################################
        # Compile and save concatenated DataFrame
        if any([args.calc_mse, args.calc_es, args.calc_crps, args.plot_ranks, args.calc_spatial_metrics, args.calc_acf,
                args.plot_power_spectrum, args.calc_quantiles]) or args.run_all:
            for var in variables:
                dfs_to_concat = []

                if args.calc_mse or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_mse[var]))

                if args.calc_es or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_es[var]))

                if args.calc_crps or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_crps[var]))
                    
                if args.plot_ranks or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_rh[var]))

                if args.calc_spatial_metrics or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_spatial_metrics[var]))
                    
                if args.calc_acf or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_acfs[var]))

                if args.plot_power_spectrum or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_psd[var]))

                if args.calc_quantiles or args.run_all:
                    dfs_to_concat.append(postproc_df(df_lists_quantiles[var]))

                if dfs_to_concat:
                    concatenated_df = pd.concat(dfs_to_concat, axis=1)
                    metrics_str = ""
                    if args.calc_mse or args.run_all:
                        metrics_str += "2"
                    if args.calc_es or args.run_all:
                        metrics_str += "3"
                    if args.calc_crps or args.run_all:
                        metrics_str += "7"
                    if args.plot_ranks or args.run_all:
                        metrics_str += "8"
                    if args.calc_spatial_metrics or args.run_all:
                        metrics_str += "10"
                    if args.calc_acf or args.run_all:
                        metrics_str += "11"
                    if args.plot_power_spectrum or args.run_all:
                        metrics_str += "12"
                    if args.calc_quantiles or args.run_all:
                        metrics_str += "17"
                    concatenated_df.to_csv(os.path.join(out_path, f"{var}/concatenated_results_{args.period}_{run_id_name}_{metrics_str}.csv"))
                    
        ##################################################################################################################
        # Compile and save concatenated DataFrame for multivariate eval
        dfs_to_concat = []

        for var_pair in variable_pairs:
            var = var_pair[0]

            if args.calc_corrs or args.run_all:
                corr_res_pair = postproc_df(df_lists_pw_corr[var_pair])
                corr_res_pair["var1"] = var_pair[0]
                corr_res_pair["var2"] = var_pair[1]
                dfs_to_concat.append(corr_res_pair)
                
            if args.rh_var_diffs or args.run_all:
                rh_diff_res_pair = postproc_df(df_lists_rh_diffs[var_pair])
                rh_diff_res_pair["var1"] = var_pair[0]
                rh_diff_res_pair["var2"] = var_pair[1]
                dfs_to_concat.append(rh_diff_res_pair)

        if dfs_to_concat:
            concatenated_df = pd.concat(dfs_to_concat, axis=0)
            metrics_str = ""
            if args.calc_corrs or args.run_all:
                metrics_str += "13"
            if args.rh_var_diffs or args.run_all:
                metrics_str += "15"
            concatenated_df.to_csv(os.path.join(out_path, f"multivariate_concatenated_results_{args.period}_{run_id_name}_{metrics_str}.csv"))
