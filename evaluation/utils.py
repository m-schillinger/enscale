import os
import re
import xarray as xr
import numpy as np
import torch
import h5py
import pdb
import cftime
import pandas as pd
import copy

def get_rcm_gcm_combinations(root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"):
    # get a dictionary with all available GCMs and RCMs
    # pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
    pattern = re.compile(r'tas_day_EUR-11_([\w-]+)_([\w-]+)_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc')

    # List all filenames in the directory
    file_names = sorted(os.listdir(root + "/train"))

    # Extract GCM-RCM combinations from file names
    rcm_gcm_combinations = []
    for file_name in file_names:
        match = pattern.match(file_name)
        if match:
            rcm_gcm_combinations.append((match.group(1), match.group(2)))

    rcm_list = [comb[0] for comb in rcm_gcm_combinations]
    gcm_list = [comb[1] for comb in rcm_gcm_combinations]
    sorted_rcms = sorted(list(set(rcm_list)))
    sorted_gcms = sorted(list(set(gcm_list)))
    rcm_dict = {sorted_rcms[i]: i for i in range(len(sorted_rcms))}
    gcm_dict = {sorted_gcms[i]: i for i in range(len(sorted_gcms))}
    return gcm_list, rcm_list, gcm_dict, rcm_dict
    
def get_run_index(rcm, gcm, rcm_list=None, gcm_list=None, root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"):
    if rcm_list is None or gcm_list is None:
        gcm_list, rcm_list, _, _ = get_rcm_gcm_combinations(root)
    rcm_gcms = [(rcm_list[i], gcm_list[i]) for i in range(len(rcm_list))]
    return np.where([rcm_gcms[i] == (rcm, gcm) for i in range(len(rcm_gcms))])[0][0]


def get_dates(rcm, gcm, root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear", mode = "train"):
    if mode == "train":
        data = xr.open_dataset(root + f"/train/pr_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc")
    elif mode == "test_interpolation":
        data = xr.open_dataset(root + f"/test/interpolation/pr_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc")
    elif mode == "test_extrapolation":
        data = xr.open_dataset(root + f"/test/extrapolation/pr_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc")
    
    if isinstance(data.time.values[0], cftime.DatetimeNoLeap):
        dates = data.indexes['time'].to_datetimeindex().strftime('%Y-%m-%d')
        dates = np.array(dates, dtype=str)
    else:
        dates = data.time.values
    return dates


def correct_units(data, data_type = "tas"):
    if data_type == "pr" and data.mean() < 0.1:
        return data * 86400 # if no normalisation, convert from kg/(m^2s) to mm/day
    elif data_type == "tas" and data.mean() > 100:
        return data - 273.15 # if no normalisation, convert from K to C
    else:
        return data

def load_data(gcm, rcm, variables, mode = "test_interpolation", run_ids = ["maybritt_3step_dense_v1"], 
              load_gan=False, load_benchmarks=True, load_diffusion=False, filter_outliers=False):
    if mode == "test_interpolation":
        dir_test_int = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/" + "interpolation"
    elif mode == "test_extrapolation":
        dir_test_int = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear/test/" + "extrapolation"
    true_data = {}
    for var in variables:
        if mode == "test_interpolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc"))[var].data, 1)
        elif mode == "test_extrapolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"))[var].data, 1)
        true_data[var] = correct_units(hr_data, var)

    gen_data = {}

    name_res = "inter" if mode == "test_interpolation" else "extra"
    run_index = get_run_index(rcm, gcm)
    for run_id in run_ids:
        gen_data[run_id] = {}
        dir_m = f"/r/scratch/groups/nm/downscaling/samples_multivariate/" + run_id
        # idx0_inter
        if run_id.endswith("dense-conv_v3"):
            try:
                data = torch.load(f"{dir_m}/idx{run_index}_{name_res}_REP.pt")
            except:
                data = torch.load(f"{dir_m}/idx{run_index}_{name_res}.pt")
        else:
            if filter_outliers:
                data = torch.load(f"{dir_m}/idx{run_index}_{name_res}_filtered.pt")
            else:
                data = torch.load(f"{dir_m}/idx{run_index}_{name_res}.pt")
        var_order = ["tas", "pr", "sfcWind", "rsds"]
        for var in variables:
            if run_id.endswith("test-rsds") or run_id.endswith("test-rsds2"):
                data_var = data[:, 0, :, :]
            else:
                data_var = data[:, var_order.index(var), :, :]
            if run_id.endswith("temporal") or run_id.endswith("temporal_v1") or run_id.endswith("temporal_v2") or run_id.endswith("temporal_v3") or run_id.endswith("temporal_v4"):
                data_var =  torch.cat([data_var[:1], data_var], dim = 0) # simple hack for now
            gen_data[run_id][var] = data_var.view(-1, 128, 128, 9).numpy()
            

    name_bench = "int" if mode == "test_interpolation" else "ext"
    if load_benchmarks:
        gen_data["analogues"] = {}
        # dir_analogues = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/analogue_result/resample_train-test-split/5-neighbours-seasonal-multivar"
        dir_analogues = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/analogue_result/resample_train-test-split/5-neighbours-seasonal-50-days-multivar"
        for var in variables:
            with h5py.File(f"{dir_analogues}/{var}_analogues_{rcm}_{gcm}_{name_bench}-period.h5", 'r') as f:
                data = f[var]
                # gen_data["analogues"][var] = np.flip(np.transpose(data[:], (0,2,3,1)), 1)
                hr_data = np.flip(np.transpose(data[:], (0,2,3,1)), 1)
                gen_data["analogues"][var] = correct_units(hr_data, var)
        
        gen_data["nn_det_per_variable"] = {}
        run_index = get_run_index(rcm, gcm)
        # dir_detnn = "/r/scratch/groups/nm/downscaling/samples/samples_det_nn_v1"
        dir_detnn = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/nn_det_per_variable/burnin24_v1"
        # tas_idx7_inter
        for var in variables:
            data = torch.load(f"{dir_detnn}/{var}_idx{run_index}_{name_res}.pt")
            gen_data["nn_det_per_variable"][var] = data.view(-1, 128, 128, 1).numpy()
        
        gen_data["idr"] = {}
        # dir_ridge = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/linear_ridge_idr"
        # dir_idr = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/nn_det_per_variable_idr"
        dir_idr = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/nn_det_per_variable_idr_v1-9samples"
        # sfcWind_idr_random_subsample_RegCM4-6_MPI-ESM-LR_ext-period
        for var in variables:
            with h5py.File(f"{dir_idr}/{var}_idr_random_subsample_{rcm}_{gcm}_{name_bench}-period.h5", 'r') as f:
                data = f[var]
                gen_data["idr"][var] = correct_units(data[:], var)

    name_res = "inter" if mode == "test_interpolation" else "extra"

    if load_gan:
        gen_data["gan"] = {}
        selected_gcm = gcm
        selected_rcm = rcm
        #pred_dir_gan = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/GAN"
        #pred_id = "010"
        pred_dir_gan = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/GAN/ensmeanMSE_content_loss"
        pred_id = "030"
        if mode == "test_interpolation":
            pred_file_name = f"pred_id-{pred_id}_sample-10_{gcm}_{rcm}_2030-2039.h5"
        elif mode == "test_extrapolation":
            pred_file_name = f"pred_id-{pred_id}_sample-10_{gcm}_{rcm}_2090-2099.h5"
        pred_file = os.path.join(pred_dir_gan, pred_file_name)
        #"pred_id-002" (univariat) sind die Vorhersagen der beiden GANs, die jeweils auf das entsprechende Target trainiert wurden
        #"pred_id-010" (multivariat) sind die Vorhersagen des einen GANs, welches auf beide Targets trainiert wurde
        for var in ["tas", "pr", "sfcWind", "rsds"]:

            with h5py.File(pred_file, 'r') as f:
                data = f[var][:]
                data = np.moveaxis(data, 1, -1)

                if pred_id in ['001', '003']:
                    print("XXX Flipping generated data along lat axis")
                    data = np.flip(data, 1)

                if (selected_gcm == 'MIROC5') and (selected_rcm == 'CCLM4-8-17'): #and (args.period in ['2030-2039', '2090-2099']):
                    # For MIROC5, CCLM4-8-17, and 2030-2039 true data has only 3650 entries
                    data = data[:true_data[variables[0]].shape[0]]

                if (selected_gcm == 'MIROC5') and (selected_rcm == 'REMO2015'):# and (args.period in ['2030-2039', '2090-2099']):
                    # For MIROC5, REMO2015, and 2030-2039 true data has only 3650 entries
                    data = data[:true_data[variables[0]].shape[0]]

                gen_data["gan"][var] = data

    if load_diffusion:
        gen_data["corrdiff"] = {}
        selected_gcm = gcm
        selected_rcm = rcm
        #pred_id = "005"
        pred_id = "020"
        pred_dir_diff = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/corrdiff"
        if mode == "test_interpolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2030-2039.nc"
        elif mode == "test_extrapolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2090-2099.nc"

        pred_file = os.path.join(pred_dir_diff, pred_file_name)

        dataset = xr.open_dataset(pred_file, group='prediction')
        for var in ["tas", "pr", "sfcWind", "rsds"]:
            data = np.array(dataset[var].values)
            data = np.moveaxis(data, 0, -1)
            gen_data["corrdiff"][var] = data
            
        gen_data["corrdiff_fixed-seeds"] = {}
        selected_gcm = gcm
        selected_rcm = rcm
        #pred_id = "005"
        pred_id = "030"
        pred_dir_diff = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/corrdiff"
        if mode == "test_interpolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2030-2039.nc"
        elif mode == "test_extrapolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2090-2099.nc"

        pred_file = os.path.join(pred_dir_diff, pred_file_name)

        dataset = xr.open_dataset(pred_file, group='prediction')
        for var in ["tas", "pr", "sfcWind", "rsds"]:
            data = np.array(dataset[var].values)
            data = np.moveaxis(data, 0, -1)
            gen_data["corrdiff_fixed-seeds"][var] = data
        
        """
        gen_data["corrdiff_mini"] = {}
        selected_gcm = gcm
        selected_rcm = rcm
        #pred_id = "005"
        pred_id = "011"
        pred_dir_diff = "/r/scratch/users/mschillinger/projects/cordex_downscaling/benchmarks/corrdiff"
        if mode == "test_interpolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2030-2039.nc"
        elif mode == "test_extrapolation":
            pred_file_name = f"corrdiff_pred_id-{pred_id}_sample-10_{selected_gcm}_{selected_rcm}_2090-2099.nc"

        pred_file = os.path.join(pred_dir_diff, pred_file_name)

        dataset = xr.open_dataset(pred_file, group='prediction')
        for var in ["tas", "pr", "sfcWind", "rsds"]:
            data = np.array(dataset[var].values)
            data = np.moveaxis(data, 0, -1)
            gen_data["corrdiff_mini"][var] = data
        """
        
    return true_data, gen_data

def load_data_counterfact(gcm, rcm, variables, mode = "interpolation", run_id = "maybritt_3step_dense-conv_v2", rcm_counterfact = None):
    dir_test_int = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-allyear/test/" + mode
    true_data = {}
    for var in variables:
        if mode == "interpolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc"))[var].data, 1)
        elif mode == "extrapolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"))[var].data, 1)
        true_data[var] = correct_units(hr_data, var)
        # remove the correct units here, as some  results are already in inherent unit

    gen_data = {}
    name_bench = "int" if mode == "interpolation" else "ext"
    name_res = "inter" if mode == "interpolation" else "extra"

    gen_data["maybritt_3step_base"] = {}
    run_index = get_run_index(rcm, gcm)
    run_index_counterfact = get_run_index(rcm_counterfact, gcm)
    dir_m = f"/r/scratch/groups/nm/downscaling/samples_multivariate/" + run_id
    # idx0_inter
    if run_index == 1 and run_index_counterfact == 4:
        data = torch.load(f"{dir_m}/idx{run_index}_{name_res}_counterfact_base.pt")    
    else:
        data = torch.load(f"{dir_m}/idx{run_index}_{name_res}_counterfact_base-for{run_index_counterfact}.pt")
    var_order = ["tas", "pr", "sfcWind", "rsds"]
    for var in variables:
        data_var = data[:, var_order.index(var), :, :]
        gen_data["maybritt_3step_base"][var] = data_var.view(-1, 128, 128, 9).numpy()
        
    gen_data["maybritt_3step_counterfact"] = {}
    dir_m = f"/r/scratch/groups/nm/downscaling/samples_multivariate/" + run_id
    # idx0_inter
    data = torch.load(f"{dir_m}/idx{run_index}_{name_res}_counterfact{run_index_counterfact}.pt")
    var_order = ["tas", "pr", "sfcWind", "rsds"]
    for var in variables:
        data_var = data[:, var_order.index(var), :, :]
        gen_data["maybritt_3step_counterfact"][var] = data_var.view(-1, 128, 128, 9).numpy()
        
    true_data_counterfact = {}
    for var in variables:
        if mode == "interpolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm_counterfact}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc"))[var].data, 1)
        elif mode == "extrapolation":
            hr_data = np.flip(xr.open_dataset(os.path.join(dir_test_int, f"{var}_day_EUR-11_{rcm_counterfact}_{gcm}_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"))[var].data, 1)
        true_data_counterfact[var] = correct_units(hr_data, var)

    return true_data, gen_data, true_data_counterfact

def add_shuffled_benchmarks(gen_data, methods = ["maybritt_2-step"], variables = ["tas", "pr", "rsds", "sfcWind"]):
    for method in methods:
        gen_data[method + "_shuffled"] = {}
        for var in variables:
            # random shuffling
            shuffled_data = copy.deepcopy(gen_data[method][var])
            np.random.shuffle(shuffled_data.transpose(3, 0, 1, 2)) # shuffle only works along the first dimension, so we need to transpose first
            shuffled_data.transpose(1,2,3,0) # transpose back
            gen_data[method + "_shuffled"][var] = shuffled_data
    return gen_data
    

def postproc_df(df_list):
    df_all = pd.concat(df_list, axis=0)
    df_all["method"] = df_all.index
    df_all.reset_index(drop=True, inplace=True)
    df_all = df_all.drop(columns=['rcm', 'gcm']).groupby("method").mean()
    return df_all

