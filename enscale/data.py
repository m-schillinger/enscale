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

# ------------- DATASET ---------------------------------------

class DownscalingDatasetNormed(Dataset):
    # similar class to DownscalingDataset, but more general for different GCM-RCM combinations
    """Most general class for datasets for downscaling.
    Requirements: ?
    """
    def __init__(self, root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                 data_types=["tas", "pr"],
                 data_types_lr = ["pr", "tas", "sfcWind", "rsds", "psl"],
                 rcm = "", gcm = "", variant = "r1i1p1",
                 one_hot = None,
                 mode = "train",
                 norm_input=None,
                 norm_output=None,
                 sqrt_transform_in=True,
                 sqrt_transform_out=True,
                 coarsened_hr=False,
                 kernel_size=1):
        """
        Parameters
        ----------
        - root: path to files
        - data_type: "tas" or "pr"
        - rcm: name of the rcm
        - gcm: name of the gcm
        - variant: variant name of the gcm run
        - return_index: If True, the get_item function returns also the index (an integer indicating which RCM is used, only needed for bias-correction)

        Notes
        ----------
        - In general, one has to pay attention with the orientation of the data: If one converts the data from a .nc file to a tensor, sometimes the first rows will correspond to South, sometimes to North.
        Therefore, we sometimes have to flip the data - we agree on "first rows = north.
        """

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
        lr_tensors = []
        for data_type in data_types:
            # loop through variables and concatenate along first dimension
            # shape in the end should be [num_variables, n_channels, image_size, image_size]
            # n_channels is 1 for now, but 2 later after fft

            # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
            hr_path = os.path.join(root, folder, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + ".nc")
            # flipping the data to have correct north-south orientation
            hr_ds = xr.open_dataset(hr_path)
            hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
            hr_data = correct_units(hr_data, data_type)
            hr_data_norm = normalise(hr_data, mode = "hr", data_type = data_type, sqrt_transform = sqrt_transform_out, norm_method = norm_output, root=root,
                                     n_keep_vals=1000, interp_step=10)            
            hr_tensors.append(hr_data_norm.unsqueeze(1))
            
        hr_data_allvars = torch.concat(hr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 128*128 or 128*128 + value_dim

        for data_type in data_types_lr:
            # low-res
            if coarsened_hr and data_type in ["tas", "pr"]:
                # fake LR input: instead of GCM data, load the coarsened HR data
                lr_path = os.path.join(root, folder, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_EUROPE_g025" +  file_suffix + ".nc")
                lr_ds = xr.open_dataset(lr_path)
            else:
                lr_path = os.path.join(root, folder, data_type + "_day_" + gcm + "_" + variant + "_rcp85_EUROPE_g025" +  file_suffix + ".nc")
                lr_ds = xr.open_dataset(lr_path)            
                
            if coarsened_hr and len(lr_ds.time) == 39813 and hr_data.shape[0] < 39813:
                # remove all 29th of February
                ds_29th = lr_ds.sel(time=lr_ds['time.day'] == 29)
                feb_29th = ds_29th.sel(time=ds_29th['time.month'] == 2)
                lr_ds = lr_ds.drop_sel(time=feb_29th.time)
                # pdb.set_trace()
                
            if len(lr_ds.time) != hr_data.shape[0]:
                print(rcm, gcm)
                print("Length mismatch. Should investigate further!")
                # pdb.set_trace()   
             
            lr_data = torch.flip(torch.from_numpy(lr_ds[data_type].data), [1])
            # data = torch.from_numpy(lr_ds[data_type].data)

            if coarsened_hr:
                # coarsened data still has nans, so replace them with zeros
                if torch.isnan(lr_data).any():
                    print("Warning: NaN values found in lr_data; replacing with zeros")
                    lr_data = torch.where(torch.isnan(lr_data), torch.zeros_like(lr_data), lr_data)
                    
            # bring lr_data to the right units
            lr_data = correct_units(lr_data, data_type)
            if kernel_size > 1:
                s1 = lr_data.shape[-2]
                s2 = lr_data.shape[-1]
                lr_data = torch.nn.functional.avg_pool2d(lr_data, kernel_size=kernel_size, stride=kernel_size)
                lr_data = torch.nn.functional.interpolate(lr_data.unsqueeze(1), size=(s1, s2), mode='nearest').squeeze(1) # upsample again to keep size
            lr_data_norm = normalise(lr_data, mode = "lr", data_type = data_type, sqrt_transform = sqrt_transform_in, norm_method = norm_input, root=root,
                                     n_keep_vals=0, interp_step=1)
            
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
        
        time_idx = torch.cat([
                time,
                doy,
                torch.sin(torch.tensor(365 / 2 / np.pi) * doy),
                torch.cos(torch.tensor(365 / 2 / np.pi) * doy),
                torch.sin(torch.tensor(365 / np.pi) * doy),
                torch.cos(torch.tensor(365 / np.pi) * doy)
            ], dim = 1)#.expand(lr_data_allvars.shape[0], lr_data_allvars.shape[1], 6)
        
        one_hot = one_hot.unsqueeze(0).expand(lr_data_allvars.shape[0], 7)
        
        self.x_data = torch.cat([lr_data_allvars.reshape(lr_data_allvars.shape[0], -1),
                   time_idx,
                   one_hot],
                    dim = -1)
        print("x data:", self.x_data.shape)
        
    def __len__(self):
        assert self.x_data.shape[0] == self.y_data.shape[0]
        return self.x_data.shape[0]
    
    def __getitem__(self, idx):
        """
        return pre-processed pair
        """
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
    
    

# -------------- NAIVE HELPERS ---------------------------------------
    
class GCMLowResRCMDataset(Dataset):
    def __init__(self,
                 ds_gcm,
                 ds_coarsened_hr):
        super(GCMLowResRCMDataset, self).__init__()
        self.ds_gcm = ds_gcm       
        self.ds_coarsened_hr = ds_coarsened_hr        
        
    def __getitem__(self, idx):
        x_coarsened, y = self.ds_coarsened_hr[idx]
        x_gcm, y = self.ds_gcm[idx]
        # pdb.set_trace()
        return x_gcm, x_coarsened[:-13], y
        
    def __len__(self):
        assert len(self.ds_gcm) == len(self.ds_coarsened_hr)
        return len(self.ds_gcm)
    
class GCMToLowResDataset(Dataset):
    def __init__(self,
                 ds_gcm,
                 ds_coarsened_hr):
        super(GCMToLowResDataset, self).__init__()
        self.ds_gcm = ds_gcm       
        self.ds_coarsened_hr = ds_coarsened_hr        
        
    def __getitem__(self, idx):
        x_coarsened, y = self.ds_coarsened_hr[idx]
        x_gcm, y = self.ds_gcm[idx]
        # pdb.set_trace()
        return x_gcm, x_coarsened[:-13]
        
    def __len__(self):
        assert len(self.ds_gcm) == len(self.ds_coarsened_hr)
        return len(self.ds_gcm)    


class GCMLowResRCMDataset_AvgPool(Dataset):
    def __init__(self,
                 ds_gcm, kernel_size = 16):
        super(GCMLowResRCMDataset_AvgPool, self).__init__()
        self.ds_gcm = ds_gcm       
        self.kernel_size = kernel_size
    def __getitem__(self, idx):
        x_gcm, y = self.ds_gcm[idx]
        x_coarsened = torch.nn.functional.avg_pool2d(y.view(-1, 128, 128), kernel_size=self.kernel_size, stride=self.kernel_size)
        return x_gcm, x_coarsened.view(y.shape[0], -1), y
        
    def __len__(self):
        return len(self.ds_gcm) 

# ------------ QUICK AND DIRTY TEMPORARY HELPER FOR HIERARCHICAL SUPER-RES ------

class DownscalingDatasetSuperNormed(Dataset):
    # class for hierarchical downscaling
    def __init__(self, root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                 data_types=["tas", "pr"],
                 data_types_lr = ["pr", "tas", "sfcWind", "rsds", "psl"],
                 rcm = "", gcm = "", variant = "r1i1p1",
                 one_hot = None,
                 mode = "train",
                 norm_input=None,
                 norm_output=None,
                 sqrt_transform_in=True,
                 sqrt_transform_out=True,
                 kernel_size=16,
                 kernel_size_middle=4,
                 mask_gcm=False,
                 one_hot_dim=7,
                 return_timepair=False,
                 clip_quantile=None, logit=False):
        """
        Parameters
        ----------
        - root: path to files
        - data_type: "tas" or "pr"
        - rcm: name of the rcm
        - gcm: name of the gcm
        - variant: variant name of the gcm run
        - return_index: If True, the get_item function returns also the index (an integer indicating which RCM is used, only needed for bias-correction)

        Notes
        ----------
        - In general, one has to pay attention with the orientation of the data: If one converts the data from a .nc file to a tensor, sometimes the first rows will correspond to South, sometimes to North.
        Therefore, we sometimes have to flip the data - we agree on "first rows = north.
        """
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
        hr_coarsened_middle_tensors = []
        lr_tensors = []
        for data_type in data_types:
            # loop through variables and concatenate along first dimension
            # shape in the end should be [num_variables, n_channels, image_size, image_size]
            # n_channels is 1 for now, but 2 later after fft

            if norm_output == "uniform":
                    # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
                if mode == "train":
                    folder_unif = "train_norm_unif"
                elif mode == "test_interpolation":
                    folder_unif = "test_norm_unif/interpolation"
                elif mode == "test_extrapolation":
                    folder_unif = "test_norm_unif/extrapolation"
                assert not sqrt_transform_out
                hr_path = os.path.join(root, folder_unif, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + "_subsample-v2.nc")
                # flipping the data to have correct north-south orientation
                hr_ds = xr.open_dataset(hr_path)
                try:
                    hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                except KeyError:
                    pdb.set_trace()                    
                hr_data_norm = hr_data.float().view(hr_data.shape[0], -1)
                if logit:
                    hr_data_norm = torch.logit(hr_data_norm)
            else:
                # example: pr_day_EUR-11_ALADIN63_MPI-ESM-LR_r1i1p1_rcp85_ALPS_cordexgrid_train-period.nc
                hr_path = os.path.join(root, folder, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_cordexgrid" +  file_suffix + ".nc")
                # flipping the data to have correct north-south orientation
                hr_ds = xr.open_dataset(hr_path)
                hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                hr_data = correct_units(hr_data, data_type)
                if data_type == "pr" and clip_quantile is not None:
                    # 
                    # hr_data = torch.clamp(hr_data, torch.quantile(hr_data, 1 - clip_quantile).item(), torch.quantile(hr_data, clip_quantile).item())
                    hr_data = batchwise_quantile_clipping(hr_data, q = clip_quantile, batch_size = 365)
                hr_data_norm = normalise(hr_data, mode = "hr", data_type = data_type, sqrt_transform = sqrt_transform_out, norm_method = norm_output, root=root,
                                        n_keep_vals=1000, interp_step=10, logit=logit)
            hr_tensors.append(hr_data_norm.unsqueeze(1))         
            
            """
            hr_coarsened_tensors_var = []
            for k in kernel_size:
                if k > 1:
                    hr_coarsened_k = torch.nn.functional.avg_pool2d(
                        hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k).view(hr_data_norm.shape[0], 1, -1)
                    hr_coarsened_tensors_var.append(hr_coarsened_k)
                else:
                    hr_coarsened_tensors_var.append(hr_data_norm.unsqueeze(1))           
            """            
            if kernel_size > 1:
                k = kernel_size
                hr_coarsened = torch.nn.functional.avg_pool2d(
                    hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k).view(hr_data_norm.shape[0], 1, -1)
                hr_coarsened_tensors.append(hr_coarsened)
            else:
                hr_coarsened_tensors.append(hr_data_norm.unsqueeze(1))   
                
            if kernel_size_middle > 1:
                k = kernel_size_middle
                hr_coarsened = torch.nn.functional.avg_pool2d(
                    hr_data_norm.view(hr_data_norm.shape[0], 128, 128).unsqueeze(1), kernel_size=k, stride=k).view(hr_data_norm.shape[0], 1, -1)
                hr_coarsened_middle_tensors.append(hr_coarsened)
            else:
                hr_coarsened_middle_tensors.append(hr_data_norm.unsqueeze(1))   
                
        hr_data_allvars = torch.concat(hr_tensors, dim = 1) # shape (n_timesteps, n_vars, spatial_dim), where spatial_dim = 128*128 or 128*128 + value_dim
        hr_data_coarsened_allvars = torch.concat(hr_coarsened_tensors, dim = 1)
        hr_data_coarsened_middle_allvars = torch.concat(hr_coarsened_middle_tensors, dim = 1)

        lr_path = os.path.join(root, folder, data_type + "_day_" + gcm + "_" + variant + "_rcp85_EUROPE_g025" +  file_suffix + ".nc")
        lr_ds = xr.open_dataset(lr_path)   
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
                
        # one_hot = one_hot.unsqueeze(0).expand(lr_data_allvars.shape[0], one_hot_dim)
        
        self.x_data = hr_data_coarsened_allvars
        print("x data:", self.x_data.shape)
        
        self.z_data = hr_data_coarsened_middle_allvars
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



# -------------- ALTERNATIVE DATASET CLASS ---------------------------------------

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
    # similar class to DownscalingDataset, but more general for different GCM-RCM combinations
    """Most general class for datasets for downscaling.
    Requirements: ?
    """
    def __init__(self, root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                 data_types=["tas", "pr"],
                 data_types_lr = ["pr", "tas", "sfcWind", "rsds", "psl"],
                 rcm = "", gcm = "", variant = "r1i1p1",
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
                 old_data=False,
                 stride_lr = None,
                 padding_lr = None,
                 filter_outliers=False,
                 precip_zeros = "random",
                 ):
        """
        Parameters
        ----------
        - root: path to files
        - data_type: "tas" or "pr"
        - rcm: name of the rcm
        - gcm: name of the gcm
        - variant: variant name of the gcm run
        - return_index: If True, the get_item function returns also the index (an integer indicating which RCM is used, only needed for bias-correction)

        Notes
        ----------
        - In general, one has to pay attention with the orientation of the data: If one converts the data from a .nc file to a tensor, sometimes the first rows will correspond to South, sometimes to North.
        Therefore, we sometimes have to flip the data - we agree on "first rows = north.
        """
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
                        print("data issues", rcm, gcm)
                        pdb.set_trace()
                    
            else:
                if not old_data:
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

                else:
                    root_old = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-DJF"
                    # example: pr_day_EUR-11_CLMcom-ETH-COSMO-crCLIM-v1-1_EC-EARTH_r1i1p1_rcp85_ALPS_DJF_cordexgrid.nc
                    hr_path = os.path.join(root_old, data_type + "_day_EUR-11_" + rcm + "_" + gcm + "_" + variant + "_rcp85_ALPS_DJF_cordexgrid.nc")
                # flipping the data to have correct north-south orientation
                hr_ds = xr.open_dataset(hr_path)
                
                if only_winter:
                    months = np.float32(hr_ds.indexes['time'].strftime("%m")).astype("int")
                    hr_ds = hr_ds.sel(time = (months == 12) | (months == 1) | (months == 2))
                
                hr_data = torch.flip(torch.from_numpy(hr_ds[data_type].data), [1])
                hr_data = hr_data.float()
                hr_data = correct_units(hr_data, data_type)
                if data_type == "pr" and clip_quantile is not None:
                    # 
                    # hr_data = torch.clamp(hr_data, torch.quantile(hr_data, 1 - clip_quantile).item(), torch.quantile(hr_data, clip_quantile).item())
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
            if not old_data:
                lr_path = os.path.join(root, folder, data_type + "_day_" + gcm + "_" + variant + "_rcp85_EUROPE_g025" +  file_suffix + ".nc")
            else:
                # example psl_day_CNRM-CM5_r1i1p1_rcp85_g025_1971-2099_EUROPE_DJF.nc
                root_old = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-DJF"
                lr_path = os.path.join(root_old, data_type + "_day_" + gcm + "_" + variant + "_rcp85_g025_1971-2099_EUROPE_DJF.nc")
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
    
def get_data(n_models=8, shuffle=True, batch_size=512, run_indices = None, variables = ["pr"], variables_lr = None, mode = "train",
             norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
             coarsened_hr=False, predict_lr=False, kernel_size=1):
    if coarsened_hr:
        root = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-allyear"
        if run_indices is None:
            run_indices = list(range(1, n_models)) # leave out one model in case of coarsened HR (model not available then)
    else:
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
        if run_indices is None:
            run_indices = list(range(n_models))
    random_state = 42
    
    if mode == "train":
        test_size = 0.1
    else:
        test_size = 0.0
    
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    
    gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
    one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
    rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
    one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
    
    if variables_lr is None:
        if coarsened_hr:
            variables_lr = ["pr", "tas"]
        else:
            variables_lr = ["pr", "tas", "sfcWind", "rsds", "psl"]
    
    print("loading data for variables:", variables)
    if predict_lr:
        if run_indices is not None:
            run_indices = [ind for ind in run_indices if ind in list(range(1, n_models))]
        else:
            run_indices = list(range(1, n_models))
        datasets_train = [GCMToLowResDataset(DownscalingDatasetNormed(root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                                                data_types=variables, 
                                                data_types_lr = variables_lr,
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                                mode=mode,
                                                norm_input=norm_input,
                                                norm_output=None,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                coarsened_hr=False,
                                                kernel_size=1),
                                             DownscalingDatasetNormed(root = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-allyear",
                                                data_types=variables, 
                                                data_types_lr = ["pr"],
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                                mode=mode,
                                                norm_input=norm_output,
                                                norm_output=None,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                coarsened_hr=True,
                                                kernel_size=kernel_size))                                             
                      for i in run_indices]

    else:
        datasets_train = [DownscalingDatasetNormed(root = root,
                                                data_types=variables, 
                                                data_types_lr = variables_lr,
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                                mode=mode,
                                                norm_input=norm_input,
                                                norm_output=norm_output,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                coarsened_hr=coarsened_hr,
                                                kernel_size=kernel_size)
                      for i in run_indices]

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
    
    return dataloader_train, dataloader_test


# ------ GET DATA 2 STEP ---------------------------------------
def get_data_2step(n_models=8, shuffle=True, batch_size=512, run_indices = None, variables = ["pr"], variables_lr = None, mode = "train",
             norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
             coarsened_hr=False, kernel_size=1):
    if coarsened_hr:
        root = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-allyear"
        if run_indices is None:
            run_indices = list(range(1, n_models)) # leave out one model in case of coarsened HR (model not available then)
    else:
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
        if run_indices is None:
            run_indices = list(range(n_models))
    random_state = 42
    
    if mode == "train":
        test_size = 0.1
    else:
        test_size = 0.0
    
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    
    gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
    one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
    rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
    one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
    
    if variables_lr is None:
        if coarsened_hr:
            variables_lr = ["pr", "tas"]
        else:
            variables_lr = ["pr", "psl","rsds", "sfcWind","tas"]
    
    print("loading data for variables:", variables)
    if run_indices is not None:
        run_indices = [ind for ind in run_indices if ind in list(range(1, n_models))]
    else:
        run_indices = list(range(1, n_models))
        
    # TO DO: will have an issue if sqrt transform in and out are not the same
    assert sqrt_transform_in == sqrt_transform_out
    datasets_train = [GCMLowResRCMDataset(DownscalingDatasetNormed(root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                                            data_types=variables, 
                                            data_types_lr = variables_lr,
                                            gcm = gcm_list[i], rcm = rcm_list[i], 
                                            one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                            mode=mode,
                                            norm_input=norm_input,
                                            norm_output=None,
                                            sqrt_transform_in=sqrt_transform_in,
                                            sqrt_transform_out=sqrt_transform_out,
                                            coarsened_hr=False,
                                            kernel_size=1),
                                            DownscalingDatasetNormed(root = "/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-allyear",
                                            data_types=variables, 
                                            data_types_lr = ["pr"],
                                            gcm = gcm_list[i], rcm = rcm_list[i], 
                                            one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                            mode=mode,
                                            norm_input=norm_output,
                                            norm_output=None,
                                            sqrt_transform_in=sqrt_transform_in,
                                            sqrt_transform_out=sqrt_transform_out,
                                            coarsened_hr=True,
                                            kernel_size=kernel_size))                                             
                    for i in run_indices]

    
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
    
    return dataloader_train, dataloader_test

def get_data_2step_naive_avg(n_models=8, shuffle=True, batch_size=512, run_indices = None, variables = ["pr"], variables_lr = None, mode = "train",
             norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
             kernel_size=1, kernel_size_hr=1, mask_gcm=False, joint_one_hot=False, return_timepair=False, ignore_one_hot_gcm = False, ignore_one_hot_rcm = False,
             tr_te_split = "random", test_model_index=None, train_model_index=None, train_run_indices=None, test_run_indices=None,
             clip_quantile=None, logit=False, normal=False, include_year=False,
             old_data=False, only_winter=False, server="ada", test_size=0.1, stride_lr=None, padding_lr=None,
             filter_outliers=False, precip_zeros="random"):

    if server == "ada":
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif server == "euler":
        root = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    if run_indices is None:
        run_indices = list(range(n_models))
    random_state = 42
    
    if mode != "train":
        test_size = 0.0
    
    if not old_data:    
        gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
        one_hot_dim1 = 7
        one_hot_dim2 = 8
    else:
        gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations_old("/r/scratch/users/mschillinger/data/downscaling/cordex/cordex-ALPS-DJF")
        one_hot_dim1 = 11
    
    if ignore_one_hot_gcm:
        assert not joint_one_hot
        
    if not joint_one_hot:
        gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
        one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
        if ignore_one_hot_gcm or tr_te_split == "gcm" or tr_te_split == "rcm_gcm":
            one_hot_gcm = torch.zeros_like(one_hot_gcm)
        print("one hot gcm:", one_hot_gcm)
        rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
        one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
        if ignore_one_hot_rcm or tr_te_split == "rcm":
            one_hot_rcm = torch.zeros_like(one_hot_rcm)
        print("one hot rcm:", one_hot_rcm)
    else:
        one_hot_run = torch.nn.functional.one_hot(torch.tensor(run_indices))
    
    if variables_lr is not None:
        assert sqrt_transform_in == sqrt_transform_out
        if not joint_one_hot:
            datasets_train = [DownscalingDatasetTwoStepNormed(root = root,
                                                    data_types=variables, 
                                                    data_types_lr = variables_lr,
                                                    gcm = gcm_list[i], rcm = rcm_list[i], 
                                                    one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                                    mode=mode,
                                                    norm_input=norm_input,
                                                    norm_output=norm_output,
                                                    sqrt_transform_in=sqrt_transform_in,
                                                    sqrt_transform_out=sqrt_transform_out,
                                                    kernel_size=kernel_size,
                                                    kernel_size_hr=kernel_size_hr,
                                                    mask_gcm=mask_gcm,
                                                    one_hot_dim=one_hot_dim1,
                                                    return_timepair=return_timepair,
                                                    clip_quantile=clip_quantile,
                                                    logit=logit,
                                                    normal=normal,
                                                    include_year=include_year,
                                                    old_data=old_data,
                                                    only_winter=only_winter,
                                                    stride_lr=stride_lr,
                                                    padding_lr=padding_lr,
                                                    filter_outliers=filter_outliers,
                                                    precip_zeros=precip_zeros)                                            
                            for i in run_indices]
        else:
            assert not old_data
            datasets_train = [DownscalingDatasetTwoStepNormed(root = root,
                                                    data_types=variables, 
                                                    data_types_lr = variables_lr,
                                                    gcm = gcm_list[i], rcm = rcm_list[i], 
                                                    one_hot = one_hot_run[i],
                                                    mode=mode,
                                                    norm_input=norm_input,
                                                    norm_output=norm_output,
                                                    sqrt_transform_in=sqrt_transform_in,
                                                    sqrt_transform_out=sqrt_transform_out,
                                                    kernel_size=kernel_size,
                                                    kernel_size_hr=kernel_size_hr,
                                                    mask_gcm=mask_gcm,
                                                    one_hot_dim=one_hot_dim2,
                                                    return_timepair=return_timepair,
                                                    clip_quantile=clip_quantile,
                                                    logit=logit,
                                                    normal=normal,
                                                    include_year=include_year,
                                                    old_data=old_data,
                                                    only_winter=only_winter,
                                                    stride_lr=stride_lr,
                                                    padding_lr=padding_lr,
                                                    filter_outliers=filter_outliers,
                                                    precip_zeros=precip_zeros)  
                            for i in run_indices]
        
    else: #DownscalingDatasetSuperNormed
        print("In get data, load Y and coarsened data on two scales.")
        print("If you want to load the normal GCM to RCM dataset, provide the variables_lr argument.")
        datasets_train = [DownscalingDatasetSuperNormed(root = root,
                                        data_types=variables, 
                                        data_types_lr = variables_lr,
                                        gcm = gcm_list[i], rcm = rcm_list[i], 
                                        one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                        mode=mode,
                                        norm_input=norm_input,
                                        norm_output=norm_output,
                                        sqrt_transform_in=sqrt_transform_in,
                                        sqrt_transform_out=sqrt_transform_out,
                                        kernel_size=kernel_size,
                                        mask_gcm=mask_gcm,
                                        one_hot_dim=7,
                                        return_timepair=return_timepair,
                                        clip_quantile=clip_quantile,
                                        logit=logit,
                                        )                                            
                for i in run_indices]
    
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
    elif tr_te_split == "gcm" or tr_te_split == "rcm" or tr_te_split == "rcm_gcm":
        if tr_te_split == "gcm":
            current_dict = gcm_dict
            current_list = gcm_list
        elif tr_te_split == "rcm":
            current_dict = rcm_dict
            current_list = rcm_list
        elif tr_te_split == "rcm_gcm":
            current_dict = gcm_dict
            current_list = gcm_list
            run_indices = [i for i in run_indices if rcm_list[i] == "RegCM4-6"]
            
        if test_model_index is None and train_model_index is None:
            raise ValueError("Supply either test_model_index or train_model_index")
        
        if train_model_index is not None and test_model_index is None:
            train_model = list(current_dict.keys())[train_model_index]
            dataset_train = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] == train_model])
            dataset_test = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] != train_model])
    
        elif test_model_index is not None and train_model_index is None:
            test_model = list(current_dict.keys())[test_model_index]
            dataset_train = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] != test_model])
            dataset_test = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] == test_model])
            
        elif test_model_index == train_model_index:
            train_model = list(current_dict.keys())[test_model_index]
            full_dataset = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] == train_model])
            train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size = test_size, random_state = random_state)
            dataset_train = Subset(full_dataset, train_indices)
            dataset_test = Subset(full_dataset, test_indices)
        
        else:
            train_model = list(current_dict.keys())[train_model_index]
            test_model = list(current_dict.keys())[test_model_index]
            dataset_train = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] == train_model])
            dataset_test = ConcatDataset([datasets_train[i] for i in run_indices if current_list[i] == test_model])    
        
        dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
        dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
        
    elif tr_te_split == "run_indices":
        dataset_train = ConcatDataset([datasets_train[i] for i in range(len(run_indices)) if run_indices[i] in train_run_indices])
        dataset_test = ConcatDataset([datasets_train[i] for i in range(len(run_indices)) if run_indices[i] in test_run_indices])
        
        print("train run indices:", train_run_indices)
        print("test run indices:", test_run_indices)
        
        # subset train dataset
        if test_size > 0:
            train_indices, _ = train_test_split(list(range(len(dataset_train))), test_size = test_size, random_state = random_state)
            dataset_train = Subset(dataset_train, train_indices)
        
        print("num train samples", len(dataset_train))
        print("num test samples", len(dataset_test))
        
        dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
        dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
        
    return dataloader_train, dataloader_test


def get_data_2step_naive_avg_multids(n_models=8, shuffle=True, batch_size=512, run_indices = None, variables = ["pr"], variables_lr = None, mode = "train",
             norm_input=None, norm_output=None, sqrt_transform_in=True, sqrt_transform_out=True,
             kernel_size=1, mask_gcm=False, joint_one_hot=False):

    root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    if run_indices is None:
        run_indices = list(range(n_models))
    random_state = 42
    
    if mode == "train":
        test_size = 0.1
    else:
        test_size = 0.0
    
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    
    if not joint_one_hot:
        gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
        one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
        rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
        one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
    
    else:
        one_hot_run = torch.nn.functional.one_hot(torch.tensor(np.arange(0,8)))
    
    if variables_lr is None:
        variables_lr = ["pr", "tas", "sfcWind", "rsds", "psl"]
        
    assert sqrt_transform_in == sqrt_transform_out
    if not joint_one_hot:
        datasets_train = [DownscalingDatasetTwoStepNormed(root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                                                data_types=variables, 
                                                data_types_lr = variables_lr,
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = torch.cat([one_hot_gcm[i], one_hot_rcm[i]]),
                                                mode=mode,
                                                norm_input=norm_input,
                                                norm_output=norm_output,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                kernel_size=kernel_size,
                                                mask_gcm=mask_gcm,
                                                one_hot_dim=7
                                                ) 
                        for i in run_indices]
    else:
        datasets_train = [DownscalingDatasetTwoStepNormed(root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                                                data_types=variables, 
                                                data_types_lr = variables_lr,
                                                gcm = gcm_list[i], rcm = rcm_list[i], 
                                                one_hot = one_hot_run[i],
                                                mode=mode,
                                                norm_input=norm_input,
                                                norm_output=norm_output,
                                                sqrt_transform_in=sqrt_transform_in,
                                                sqrt_transform_out=sqrt_transform_out,
                                                kernel_size=kernel_size,
                                                mask_gcm=mask_gcm,
                                                one_hot_dim=8
                                                )                                            
                        for i in run_indices]
    
    dataloaders_train = []
    dataloaders_test = []
    if test_size > 0:
        for i in run_indices:
            train_indices, test_indices = train_test_split(list(range(len(datasets_train[i]))), test_size = test_size, random_state = random_state)
            dataset_train = Subset(datasets_train[i], train_indices)
            dataset_test = Subset(datasets_train[i], test_indices)
            dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
            dataloaders_train.append(dataloader_train)
            dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
            dataloaders_test.append(dataloader_test)
    else:
        for i in run_indices:
            dataloader_train = DataLoader(datasets_train[i], batch_size, shuffle=shuffle)
            dataloaders_train.append(dataloader_train)
            dataloader_test = None
    
    return dataloaders_train, dataloaders_test