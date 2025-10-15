import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import ConcatDataset
from torch.distributions.normal import Normal
import os
import numpy as np
import re
import xarray as xr
from utils import *
import scipy

# -------------- DATASET CLASS ---------------------------------------

def get_normed_data(lr_ds, data_type, norm_input, sqrt_transform_in, root, mask_gcm):
    lr_data = torch.flip(torch.from_numpy(lr_ds[data_type].data), [1])
    # data = torch.from_numpy(lr_ds[data_type].data)
    # bring lr_data to the right units
    lr_data = correct_units(lr_data, data_type)
    lr_data_norm = normalise(lr_data, mode = "lr", data_type = data_type, sqrt_transform = sqrt_transform_in, norm_method = norm_input, root=root,
                                n_keep_vals=0, interp_step=1)

    if mask_gcm:
        mask_r = torch.ones(20)
        mask_r[5:14] = 0
        mask_c = torch.ones(36)
        mask_c[10:26] = 0
        lr_data_norm = lr_data_norm.view(lr_data_norm.shape[0], 20, 36)
        lr_data_norm[:, mask_r == 1, :][:, :, mask_c == 1] = -1
        lr_data_norm = lr_data_norm.view(lr_data_norm.shape[0], -1)
        
    return lr_data_norm

class DownscalingDatasetTwoStepNormed(Dataset):
    """
    Downscaling data loading class for two-step downscaling (GCM to coarse RCM to fine RCM).
    Loads data from local .nc files, normalises it and returns pairs of low-res and high-res data.
    1. Loads low-res GCM data and normalises it
    2. Loads high-res RCM data and normalises it
    3. Coarsens high-res RCM data to coarse RCM resolution (by average pooling)
    4. Returns (low-res GCM data, coarsened high-res RCM data) as input and high-res RCM data as output
    5. Optionally, appends time information (day of year) and one-hot encoding of GCM and RCM to the input
    
    Returns either:
    - (x, z, y) where x is the low-res GCM data + time + one-hot, z is the coarsened high-res RCM data, y is the high-res RCM data
    - (x, z, y, x_next, z_next, y_next) if return_timepair=True, where x_next, z_next, y_next are the data for the next time step (used for temporal consistency training)
    
    Parameters
    ----------
    - root: path to files
    - data_types: list of variables to load for high-res RCM data (e.g. ["pr", "tas"])
    - data_types_lr: list of variables to load for low-res GCM data (e.g. ["pr", "tas", "sfcWind", "rsds", "psl"])
    - rcm: name of the rcm
    - gcm: name of the gcm
    - variant: variant name of the gcm run
    - one_hot: one-hot encoding tensor for the (gcm, rcm) pair, shape (one_hot_dim,)
    - mode: "train", "test_interpolation" or "test_extrapolation" (used only to find correct folder for data loading)
    - norm_input: normalisation method for low-res GCM data
    - norm_output: normalisation method for high-res RCM data
    - sqrt_transform_in: if True, apply square-root transform to low-res GCM data
    - sqrt_transform_out: if True, apply square-root transform to high-res RCM data
    - kernel_size: kernel size for average pooling to coarsen high-res RCM data
    - kernel_size_hr: kernel size for additional coarsening of the target high-res RCM data (used in intermediate steps of enscale's super-resolution model)
    - mask_gcm: if True, mask out ocean points in low-res GCM data (set to -1)
    - one_hot_dim: dimension of the one-hot encoding (7 if separate GCM and RCM one-hot, 8 if joint one-hot for each (GCM, RCM) pair)
    - return_timepair: if True, return also the data for the next time step (used for temporal consistency training)
    - clip_quantile: if not None, clip high-res RCM data to the given quantile
    - logit: if True, apply logit transform to high-res RCM data (only useful if norm_output is "uniform" or "uniform_per_model")
    - normal: if True, apply normal transform to high-res RCM data (only useful norm_output is "uniform" or "uniform_per_model")
    - include_year: if True, include year as input feature (in addition to day of year)
    - levels: list of pressure levels to load for 3D variables (e.g. "hus", "ua", "va", "zg")
    - only_winter: if True, load only data from December, January, February
    - stride_lr: stride for average pooling of low-res GCM data (default: kernel_size)
    - padding_lr: padding for average pooling of low-res GCM data (default: 0)
    - filter_outliers: if True, load data from outlier-filtered files (only for high-res RCM data; currently filtering is done separately and saved on disk, only for "pr" and "rsds")
    - precip_zeros: how to treat zeros in precipitation data when using norm_output="uniform" or "uniform_per_model"; also done separately and saved on disk (options: "random", "random_correlated", "constant")
    """
    def __init__(self, 
                 root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                 data_types=["tas", "pr", "sfcWind", "rsds"],
                 data_types_lr = ["pr", "tas", "sfcWind", "rsds", "psl"],
                 rcm = "", 
                 gcm = "", 
                 variant = "r1i1p1",
                 one_hot = None,
                 mode = "train",
                 norm_input=None,
                 norm_output=None,
                 sqrt_transform_in=True,
                 sqrt_transform_out=True,
                 # kernel_size=[16],
                 kernel_size=16,
                 kernel_size_hr=1,
                 mask_gcm=False,
                 one_hot_dim=7,
                 return_timepair=False,
                 clip_quantile=None, 
                 logit=False,
                 normal=False,
                 include_year=False,
                 levels = [50000, 85000],
                 only_winter = False,
                 stride_lr = None,
                 padding_lr = None,
                 filter_outliers=False,
                 precip_zeros = "random",
                 ):
        """
        Notes
        - In general, one has to pay attention with the orientation of the data: If one converts the data from a .nc file to a tensor, sometimes the first rows will correspond to South, sometimes to North.
        Therefore, we sometimes have to flip the data - we agree on "first rows = north.
        """
        print("Loading data for RCM:", rcm, "GCM:", gcm)
        self.return_timepair = return_timepair
        if mode == "train":
            file_suffix = "_train-period"
            folder = "train"
        elif mode == "test_interpolation":
            file_suffix = "_2030-2039"
            folder = "test/interpolation"
        elif mode == "test_extrapolation":
            file_suffix = "_2090-2099"
            folder = "test/extrapolation"
        else:
            raise ValueError("In DownscalingDatasetNormed: mode not recognised")
        
        hr_tensors = []
        hr_coarsened_tensors = []
        lr_tensors = []
        for data_type in data_types:
            # loop through variables and concatenate along first dimension
            # shape in the end should be [num_variables, n_channels, image_size, image_size]
            # n_channels is 1 for now, but 2 later after fft
            
            if norm_output == "uniform_per_model" or norm_output == "uniform":           
                    # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
                if mode == "train":
                    folder_unif = "train_norm_unif"
                elif mode == "test_interpolation":
                    folder_unif = "test_norm_unif/interpolation"
                elif mode == "test_extrapolation":
                    folder_unif = "test_norm_unif/extrapolation"
                assert not sqrt_transform_out
                if norm_output == "uniform_per_model":
                    assert not filter_outliers
                    if data_type == "pr":
                        #pr_day_EUR-11_CCLM4-8-17_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period_per-model-full-period_noisy.nc
                        # pr_day_EUR-11_CCLM4-8-17_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_train-period_per-model-full-period_noisy.nc
                        hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_per-model-full-period_noisy.nc")
                    else:
                        #tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_train-period_subsample-per-model.nc
                        hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-per-model.nc")    
                else: 
                    if data_type == "pr":
                        if not filter_outliers:
                            if precip_zeros == "random_correlated":
                                hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2-random-correlated.nc")
                            elif precip_zeros == "random":
                                hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2-random.nc")
                            elif precip_zeros == "constant":
                                hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2.nc")
                        else:
                            hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2-random-filtered.nc")
                    else:
                        if not filter_outliers or data_type == "tas" or data_type == "sfcWind":
                            hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2.nc")
                        else:
                            # rsds and filter outliers
                            hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2-filtered.nc")
                # flipping the data to have correct north-south orientation
                hr_ds = xr.open_dataset(hr_path)
                
                if only_winter:
                    months = np.float32(hr_ds.indexes['time'].strftime("%m")).astype("int")
                    hr_ds = hr_ds.sel(time = (months == 12) | (months == 1) | (months == 2))
                
                hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                hr_data_norm = hr_data.float().view(hr_data.shape[0], -1)
                if logit:
                    hr_data_norm = torch.logit(hr_data_norm)
                elif normal:
                    #hr_data_norm_notransf = torch.clone(hr_data_norm)
                    hr_np = hr_data_norm.detach().cpu().numpy()
                    hr_np_gauss = scipy.stats.norm.ppf(hr_np) # more stable than torch.Normal.icdf
                    hr_data_norm = torch.from_numpy(hr_np_gauss).to(hr_data_norm.dtype).to(hr_data_norm.device)

                    if torch.any(torch.isnan(hr_data_norm)) or torch.any(torch.isinf(hr_data_norm)):
                        print("data issues (have nans)", rcm, gcm)
                    
            else:
                # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
                if not filter_outliers or data_type == "tas" or data_type == "sfcWind":
                    hr_path = os.path.join (root, folder, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + ".nc")
                else:
                    if mode == "train":
                        folder_filtered = "train_outlier_filtered"
                    elif mode == "test_interpolation":
                        folder_filtered = "test_outlier_filtered/interpolation"
                    elif mode == "test_extrapolation":
                        folder_filtered = "test_outlier_filtered/extrapolation"
                    hr_path = os.path.join(root, folder_filtered, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_filtered.nc")

                # flipping the data to have correct north-south orientation
                hr_ds = xr.open_dataset(hr_path)
                
                if only_winter:
                    months = np.float32(hr_ds.indexes['time'].strftime("%m")).astype("int")
                    hr_ds = hr_ds.sel(time = (months == 12) | (months == 1) | (months == 2))
                
                hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                hr_data = hr_data.float()
                hr_data = correct_units(hr_data, data_type)
                if data_type == "pr" and clip_quantile is not None:
                    # hr_data = torch.clamp(hr_data, torch.quantile(hr_data, 1 - clip_quantile).item(), torch.quantile(hr_data, clip_quantile).item())
                    # changed to batchwise clipping to shorten preprocessing time
                    hr_data = batchwise_quantile_clipping(hr_data, q = clip_quantile, batch_size = 365)
                    
                hr_data_norm = normalise(hr_data, mode = "hr", data_type = data_type, sqrt_transform = sqrt_transform_out, norm_method = norm_output, root=root,
                                        n_keep_vals=1000, interp_step=10, logit=logit, normal=normal)
                # if numpy array, convert to torch tensor again
                if isinstance(hr_data_norm, np.ndarray):
                    hr_data_norm = torch.from_numpy(hr_data_norm)
                
            if kernel_size_hr > 1:
                hr_coarsened = torch.nn.functional.avg_pool2d(
                    hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=kernel_size_hr, stride=kernel_size_hr).view(hr_data_norm.shape[0], 1, -1)
                hr_tensors.append(hr_coarsened)
            else:
                hr_tensors.append(hr_data_norm.unsqueeze(1))
            
            if kernel_size > 1:
                k = kernel_size
                if stride_lr is None:
                    stride_lr = k
                if padding_lr is None:
                    padding_lr = 0
                hr_coarsened = torch.nn.functional.avg_pool2d(
                    # hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k//2, padding=k//2).view(hr_data_norm.shape[0], 1, -1)
                    # hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k//2, padding=0).view(hr_data_norm.shape[0], 1, -1)
                    hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=stride_lr, padding=padding_lr).view(hr_data_norm.shape[0], 1, -1)
                hr_coarsened_tensors.append(hr_coarsened)
            else:
                hr_coarsened_tensors.append(hr_data_norm.unsqueeze(1))   

        hr_data_allvars = torch.concat(hr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 128*128 or 128*128 + value_dim
        hr_data_coarsened_allvars = torch.concat(hr_coarsened_tensors, dim = 1)

        for data_type in data_types_lr:
            # low-res
            lr_path = os.path.join(root, folder, data_type + "_day_" + gcm + "_" + variant + "_rcp85_EUROPE_g025" +  file_suffix + ".nc")
            lr_ds = xr.open_dataset(lr_path)            
                
            if only_winter:
                months = np.float32(lr_ds.indexes['time'].strftime("%m")).astype("int")
                lr_ds = lr_ds.sel(time = (months == 12) | (months == 1) | (months == 2))
                
            if len(lr_ds.time) != hr_data.shape[0]:
                print(rcm, gcm)
                print("Length mismatch. Should investigate further!")
                # pdb.set_trace()   
            
            if data_type in ["hus", "ua", "va", "zg"]:
                for level in levels:
                    lr_ds_level = lr_ds.sel(plev=level)
                    lr_data_norm = get_normed_data(lr_ds_level, data_type, norm_input, sqrt_transform_in, root, mask_gcm)
                    lr_tensors.append(lr_data_norm.unsqueeze(1)) 
            else:
                lr_data_norm = get_normed_data(lr_ds, data_type, norm_input, sqrt_transform_in, root, mask_gcm)
                lr_tensors.append(lr_data_norm.unsqueeze(1))             

        lr_data_allvars = torch.concat(lr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 20*36 or 20*36 + value_dim        
        months_np = np.float32(lr_ds.indexes['time'].strftime("%m")).astype("int")
        days_np = np.float32(lr_ds.indexes['time'].strftime("%d")).astype("int")
        years_np = np.float32(lr_ds.indexes['time'].strftime("%Y"))
        time = torch.from_numpy(years_np).unsqueeze(1)
        is_leap = is_leap_year(years_np)
        leap_year_mask = is_leap & (months_np == 2) & (days_np == 29)
        consider_leap = True if np.any(leap_year_mask) else False
        doy = torch.from_numpy(day_of_year_vectorized(months_np, days_np, is_leap, consider_leap=consider_leap)).unsqueeze(1)
        
        # compute the X, Y pairs        
        self.y_data = hr_data_allvars # torch.reshape(hr_data_allvars, (hr_data_allvars.shape[0], -1))
        print("y data:", self.y_data.shape)
        
        if include_year:
            time_idx = torch.cat([
                    time, # remove time to enable extrapolation
                    doy,
                    torch.sin(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.sin(torch.tensor(365 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / np.pi) * doy)
                ], dim = 1)#.expand(lr_data_allvars.shape[0], lr_data_allvars.shape[1], 6)
        else:
            time_idx = torch.cat([
                    # time, # remove time to enable extrapolation
                    doy,
                    torch.sin(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / 2 / np.pi) * doy),
                    torch.sin(torch.tensor(365 / np.pi) * doy),
                    torch.cos(torch.tensor(365 / np.pi) * doy)
                ], dim = 1)#.expand(lr_data_allvars.shape[0], lr_data_allvars.shape[1], 6)

        
        one_hot = one_hot.unsqueeze(0).expand(lr_data_allvars.shape[0], one_hot_dim)
        
        self.x_data = torch.cat([lr_data_allvars.reshape(lr_data_allvars.shape[0], -1),
                   time_idx,
                   one_hot],
                    dim = -1)
        print("x data:", self.x_data.shape)
        
        self.z_data = hr_data_coarsened_allvars
        print("z (x coarsened) data:", self.z_data.shape)
        
    def __len__(self):
        assert self.x_data.shape[0] == self.y_data.shape[0]
        assert self.x_data.shape[0] == self.z_data.shape[0]
        if self.return_timepair:
            return self.x_data.shape[0] - 1
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        """
        return pre-processed pair
        """
        if self.return_timepair:
            x = self.x_data[idx]
            z = self.z_data[idx]
            y = self.y_data[idx]
            x_next = self.x_data[idx + 1]
            y_next = self.y_data[idx + 1]
            z_next = self.z_data[idx + 1]
            return x, z, y, x_next, z_next, y_next
        else:
            x = self.x_data[idx]
            y = self.y_data[idx]
            z = self.z_data[idx]
            return x, z, y




# ------------ GET DATA ---------------------------------------

def get_data_2step_naive_avg(n_models=8, shuffle=True, batch_size=512, run_indices = None, 
                            joint_one_hot=False, ignore_one_hot_gcm = False, ignore_one_hot_rcm = False,
                            tr_te_split = "random", test_size=0.1,
                            server="ada",
                            variables = ["pr"], variables_lr = None, mode = "train",
                            norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
                            kernel_size=1, kernel_size_hr=1, mask_gcm=False, return_timepair=False,
                            clip_quantile=None, logit=False, normal=False, include_year=False,
                            only_winter=False, stride_lr=None, padding_lr=None,
                            filter_outliers=False, precip_zeros="random",):
    """
    Get data loaders for two-step downscaling (GCM to coarse RCM to fine RCM).
    Wrapper around DownscalingDatasetTwoStepNormed.
    
    Parameters:
    - n_models: number of (GCM, RCM) combinations available; not used if run_indices is provided
    - shuffle: whether to shuffle the training data in the dataloader
    - batch_size: batch size for the dataloader
    - run_indices: list of indices of (GCM, RCM) combinations to use; if None, the first n_models combinations are used
    - joint_one_hot: if True, use a joint one-hot encoding for each (GCM, RCM) pair; if False, use separate one-hot encodings for GCM and RCM
    - ignore_one_hot_gcm: if True, ignore the one-hot encoding for GCM (set to zero); only used if joint_one_hot is False
    - ignore_one_hot_rcm: if True, ignore the one-hot encoding for RCM (set to zero); only used if joint_one_hot is False
    - tr_te_split: method to split training and testing data; only "random" is currently included
    - test_size: fraction of data to use as test set
    - server: "ada" or "euler"; determines the root path for data loading
    - variables, variables_lr, mode, norm_input, norm_output, sqrt_transform_in, sqrt_transform_out,
      kernel_size, kernel_size_hr, mask_gcm, return_timepair,
      clip_quantile, logit, normal, include_year,
      only_winter, stride_lr, padding_lr,
      filter_outliers, precip_zeros: parameters passed to DownscalingDatasetTwoStepNormed
    """
    if server == "ada":
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif server == "euler":
        root = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    if run_indices is None:
        run_indices = list(range(n_models))
    random_state = 42
    
    if mode != "train":
        test_size = 0.0
    
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    one_hot_dim1 = 7
    one_hot_dim2 = 8
    
    if ignore_one_hot_gcm:
        assert not joint_one_hot
        
    if not joint_one_hot:
        gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
        one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
        if ignore_one_hot_gcm or tr_te_split == "gcm" or tr_te_split == "rcm_gcm":
            one_hot_gcm = torch.zeros_like(one_hot_gcm)
        rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
        one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
        if ignore_one_hot_rcm or tr_te_split == "rcm":
            one_hot_rcm = torch.zeros_like(one_hot_rcm)
    else:
        one_hot_run = torch.nn.functional.one_hot(torch.tensor(run_indices))
    
    if variables_lr is not None:
        assert sqrt_transform_in == sqrt_transform_out
        if not joint_one_hot:
            one_hot = torch.cat([one_hot_gcm, one_hot_rcm], dim = 1)
            one_hot_dim = one_hot_dim1
        else:
            one_hot = one_hot_run
            one_hot_dim = one_hot_dim2
            
        datasets_train = [DownscalingDatasetTwoStepNormed(root = root,
                                                data_types=variables, 
                                                data_types_lr = variables_lr,
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = one_hot[i],
                                                mode=mode,
                                                norm_input=norm_input,
                                                norm_output=norm_output,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                kernel_size=kernel_size,
                                                kernel_size_hr=kernel_size_hr,
                                                mask_gcm=mask_gcm,
                                                one_hot_dim=one_hot_dim,
                                                return_timepair=return_timepair,
                                                clip_quantile=clip_quantile,
                                                logit=logit,
                                                normal=normal,
                                                include_year=include_year,
                                                only_winter=only_winter,
                                                stride_lr=stride_lr,
                                                padding_lr=padding_lr,
                                                filter_outliers=filter_outliers,
                                                precip_zeros=precip_zeros)                                            
                        for i in run_indices]
        
    else: 
        raise NotImplementedError("Please provide variables_lr")
    
    if tr_te_split == "random":
        # REMOVED all RCMs except ALADIN
        # full_dataset = ConcatDataset([datasets_train[i] for i in run_indices if rcm_list[i] == "ALADIN63"])
        full_dataset = ConcatDataset(datasets_train)
        
        if test_size > 0:
            train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size = test_size, random_state = random_state)
            dataset_train = Subset(full_dataset, train_indices)
            dataset_test = Subset(full_dataset, test_indices)
            dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
            dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
        else:
            dataloader_train = DataLoader(full_dataset, batch_size, shuffle=shuffle)
            dataloader_test = None
    else:
        raise NotImplementedError("Please use tr_te_split=random, other options have been removed for simplicity.")
        
    return dataloader_train, dataloader_test