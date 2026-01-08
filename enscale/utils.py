import os
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy
import yaml
from dataclasses import asdict

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_config_to_file(cfg, save_path, filename="config.yaml"):
    os.makedirs(save_path, exist_ok=True)

    path = os.path.join(save_path, filename)
    with open(path, "w") as f:
        yaml.safe_dump(
            asdict(cfg),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def is_leap_year(year):
    return np.where((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), True, False)
    
def day_of_year_vectorized(months, days, leap_years, consider_leap=True):
    """
    Return the day of the year for a given list of months, days, and years.

    Parameters:
    - months: NumPy array or list of integers representing the months (1-12).
    - days: NumPy array or list of integers representing the days of the month.
    - years: Boolean array indicating which years are leap years (can get for an xarray via ds.time.dt.is_leap_year)
    - consider_leap: Boolean indicating whether to consider leap years (default is True).

    Returns:
    - NumPy array of integers representing the day of the year.
    """
    days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    # Calculate day of the year
    day_of_year = np.cumsum(days_in_month)[months - 1] + days

    # Adjust for leap year
    if consider_leap:
        day_of_year[leap_years & (months > 2)] += 1

    return day_of_year

def correct_units(data, data_type = "tas"):
    if data_type == "pr" and data.mean() < 0.1:
        return data * 86400 # if no normalisation, convert from kg/(m^2s) to mm/day
    elif data_type == "tas" and data.mean() > 100:
        return data - 273.15 # if no normalisation, convert from K to C
    else:
        return data
    
def get_rcm_gcm_combinations(root):
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

# ---- HELPERS FOR RUN INDEX -----  
def get_run_index(rcm, gcm, rcm_list=None, gcm_list=None, root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"):
    if rcm_list is None or gcm_list is None:
        gcm_list, rcm_list, _, _ = get_rcm_gcm_combinations(root)
    rcm_gcms = [(rcm_list[i], gcm_list[i]) for i in range(len(rcm_list))]
    return np.where([rcm_gcms[i] == (rcm, gcm) for i in range(len(rcm_gcms))])[0][0]


def get_run_index_from_onehot(
    one_hot_vec,
    gcm_dict,
    rcm_dict,
    gcm_list=None,
    rcm_list=None,
    root="/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
):
    """
    Convert a one-hot encoding (concatenated GCM + RCM) to the run index
    using get_run_index.
    
    one_hot_vec: torch.Tensor [BS, num_gcm + num_rcm]
    gcm_dict, rcm_dict: dicts from names -> indices
    gcm_list, rcm_list: optional, else loaded from root
    """
    if gcm_list is None or rcm_list is None:
        gcm_list, rcm_list, _, _ = get_rcm_gcm_combinations(root)

    num_gcm = len(gcm_dict)
    num_rcm = len(rcm_dict)

    gcm_idx = one_hot_vec[:, :num_gcm].argmax(dim=1).cpu().numpy()
    rcm_idx = one_hot_vec[:, num_gcm:num_gcm+num_rcm].argmax(dim=1).cpu().numpy()

    gcm_names = [list(gcm_dict.keys())[list(gcm_dict.values()).index(i)] for i in gcm_idx]
    rcm_names = [list(rcm_dict.keys())[list(rcm_dict.values()).index(i)] for i in rcm_idx]
    
    run_indices = [get_run_index(r, g, rcm_list, gcm_list, root) for r, g in zip(rcm_names, gcm_names)]
    
    return np.array(run_indices)


# ----------- ADD ONE HOT ENCODING TO MODEL INPUT ----
def add_one_hot(xc, one_hot_in_super=False, conv=False, x=None, one_hot_dim=0):
    if one_hot_in_super and conv:
        one_hot = x[:, -one_hot_dim:]  # Shape: (BS, one_hot_dim)
        one_hot_channels = one_hot.unsqueeze(-1).expand(-1, -1, xc.shape[2])  # Shape: (BS, one_hot_dim, image_size²)
        xc = torch.cat([xc, one_hot_channels], dim=1)  # Shape: (BS, n_channels + one_hot_dim, image_size²)

    elif one_hot_in_super and not conv:
        xc = torch.cat([xc, x[:, -one_hot_dim:]], dim=1)
    return xc

# ---------- NORMALISATION HELPERS -----------------------------------

def apply_variable_transform(x, var, sqrt_cfg):
    if sqrt_cfg.get(var, False):
        return torch.sqrt(x)
    return x

PRIMITIVE_STATS = {
    "tas": (10.0, 10.0),
    "pr": (0.0, 1.0),
    "rsds": (150.0, 100.0),
    "sfcWind": (2.2, 0.6),
    "psl": (1e5, 1e3),
}

def primitive_normalise(x, var):
    mean, std = PRIMITIVE_STATS[var]
    return (x - mean) / std


def primitive_unnormalise(x, var):
    """Unnormalise data using primitive statistics."""
    mean, std = PRIMITIVE_STATS[var]
    return x * std + mean


def reverse_variable_transform(x, var, sqrt_cfg):
    """Reverse variable transforms (e.g., undo sqrt transform)."""
    if sqrt_cfg.get(var, False):
        return x ** 2
    return x


def get_mode_from_kernel_size(kernel_size_hr):
    """Map kernel_size_hr to mode string for unnormalisation."""
    if kernel_size_hr == 1:
        return "hr"
    elif kernel_size_hr == 2:
        return "hr_avg_2"
    elif kernel_size_hr == 4:
        return "hr_avg_4"
    elif kernel_size_hr == 8:
        return "hr_avg_8"
    elif kernel_size_hr == 16:
        return "hr_avg"
    elif kernel_size_hr == 32:
        return "hr_avg_32"
    elif kernel_size_hr == 64:
        return "hr_avg_64"
    else:
        raise ValueError(f"Unknown kernel_size_hr: {kernel_size_hr}")

def load_norm_stats(cfg, mode, var, suffix, device=None):
    pattern = cfg.stats.pattern[cfg.method]
    path = os.path.join(
        cfg.stats.root,
        pattern.format(
            mode=mode,
            var=var,
            suffix=suffix,
        )
    )
    stats = torch.load(path, map_location=device)
    return stats

def load_norm_stats_for_variables(cfg, variables, kernel_size_hr, data_dir, device=None):
    """Load normalisation statistics for all variables.
    
    Args:
        cfg: Preprocessing config object with sqrt_transform and normalisation settings
        variables: List of variable names
        kernel_size_hr: Kernel size for high resolution (determines mode)
        data_dir: Root directory containing norm_stats
        device: Device to load tensors to
        
    Returns:
        Dictionary mapping variable names to norm_stats tensors (or None if not applicable)
    """
    norm_stats_dict = {}
    mode = get_mode_from_kernel_size(kernel_size_hr)
    
    # Only load norm stats for HR full resolution (kernel_size_hr == 1)
    if kernel_size_hr != 1:
        for var in variables:
            norm_stats_dict[var] = None
        return norm_stats_dict
    
    for var in variables:
        # Determine if this variable uses sqrt transform
        suffix = "_sqrt" if cfg.sqrt_transform.get(var, False) else ""
        
        norm_stats_dict[var] = load_norm_stats(
            cfg.preprocessing,
            mode,
            var,
            suffix,
            device=device,
        )
    
    return norm_stats_dict

def ecdf_normalise(x, norm_stats, len_full_data=int(3e4)): # TO DO: need to make more flexible
    data_norm = torch.zeros_like(x)
    probs = torch.linspace(1, len_full_data, len_full_data) / (len_full_data + 1) 
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            quantiles = norm_stats[:, i, j]
            data_norm[:, i, j] = torch.tensor(np.interp(x[:, i, j].detach().cpu().numpy(), quantiles.detach().cpu().numpy(), probs))
    return data_norm

def normalise(
    data,
    *,
    var,
    mode,
    cfg,
    norm_stats=None,
):
    # 1. Variable transforms
    data = apply_variable_transform(data, var, cfg.sqrt_transform)

    suffix = "_sqrt" if cfg.sqrt_transform.get(var, False) else ""

    # 2. Normalisation
    if not cfg.normalisation.apply or cfg.normalisation.method == "none":
        data_norm = data

    elif cfg.normalisation.method == "primitive":
        data_norm = primitive_normalise(data, var)

    elif cfg.normalisation.method in {"normalise_pw", "normalise_scalar"}:
        if norm_stats is None:
            norm_stats = load_norm_stats(
                cfg.normalisation,
                mode,
                var,
                suffix,
                device=data.device if torch.is_tensor(data) else None,
            )
        data_norm = (data - norm_stats["mean"]) / norm_stats["std"]

    elif cfg.normalisation.method == "uniform":
        if norm_stats is None:
            norm_stats = load_norm_stats(
                cfg.normalisation,
                mode,
                var,
                suffix,
            )
        data_norm = ecdf_normalise(data, norm_stats)

    else:
        raise ValueError(f"Unknown norm method: {cfg.normalisation.method}")

    # 3. Flatten (always last)
    data_norm = data_norm.reshape(data_norm.shape[0], -1)

    # 4. Post transforms
    if cfg.post_transform.logit:
        data_norm = torch.logit(data_norm)
    if cfg.post_transform.gaussian:
        data_np = data_norm.detach().cpu().numpy()
        data_norm = torch.from_numpy(scipy.stats.norm.ppf(data_np)).to(data_norm.dtype).to(data_norm.device)

    return data_norm

def unnormalise(
    data_norm,
    *,
    var,
    mode,
    s1,
    s2,
    cfg,
    norm_stats=None,
):
    """Unnormalise data - inverse of normalise function.
    
    Args:
        data_norm: Normalised data (flattened to shape [batch, s1*s2])
        var: Variable name (e.g., "tas", "pr", "rsds")
        s1: Spatial dimension 1 (height)
        s2: Spatial dimension 2 (width)
        cfg: Configuration object with sqrt_transform, normalisation, and post_transform settings
        norm_stats: Pre-loaded normalisation statistics (optional)
    
    Returns:
        Unnormalised data reshaped to [batch, s1, s2]
    """
    # 1. Reverse post transforms
    data_unnorm = data_norm.clone()
    
    if cfg.post_transform.logit:
        data_unnorm = torch.sigmoid(data_unnorm)
    if cfg.post_transform.gaussian:
        data_np = data_unnorm.detach().cpu().numpy()
        data_unnorm = torch.from_numpy(scipy.stats.norm.cdf(data_np)).to(data_unnorm.dtype).to(data_unnorm.device)

    # 2. Unnormalisation
    if not cfg.normalisation.apply or cfg.normalisation.method == "none":
        data = data_unnorm

    elif cfg.normalisation.method == "primitive":
        data = primitive_unnormalise(data_unnorm, var)

    elif cfg.normalisation.method in {"normalise_pw", "normalise_scalar"}:
        if norm_stats is None:
            suffix = "_sqrt" if cfg.sqrt_transform.get(var, False) else ""
            norm_stats = load_norm_stats(
                cfg.normalisation,
                mode,
                var,
                suffix,
                device=data_unnorm.device if torch.is_tensor(data_unnorm) else None,
            )
        # move norm stats to the same device as data
        device = data_unnorm.device
        mean = norm_stats["mean"].to(device)
        std = norm_stats["std"].to(device)
        data = data_unnorm * std + mean

    elif cfg.normalisation.method == "uniform":
        if norm_stats is None:
            suffix = "_sqrt" if cfg.sqrt_transform.get(var, False) else ""
            norm_stats = load_norm_stats(
                cfg.normalisation,
                mode,
                var,
                suffix,
            )
        
        len_full_data = norm_stats.shape[0] if isinstance(norm_stats, torch.Tensor) else norm_stats.shape[0]
        probs = torch.linspace(1, len_full_data, len_full_data) / (len_full_data + 1)
        
        data = torch.zeros(data_unnorm.shape[0], s1, s2)
        data_unnorm_reshaped = data_unnorm.view(data_unnorm.shape[0], s1, s2)
        
        for i in range(s1):
            for j in range(s2):
                quantiles = norm_stats[:, i, j]
                data[:, i, j] = torch.tensor(np.interp(
                    data_unnorm_reshaped[:, i, j].detach().cpu().numpy(), 
                    probs, 
                    quantiles.detach().cpu().numpy()
                ))

    else:
        raise ValueError(f"Unknown norm method: {cfg.normalisation.method}")

    # 3. Reshape to spatial dimensions
    data_reshaped = data.view(data.shape[0], s1, s2) if not torch.is_tensor(data) or len(data.shape) == 1 else (
        data if len(data.shape) == 3 else data.view(data.shape[0], s1, s2)
    )

    # 4. Reverse variable transforms
    data_reshaped = reverse_variable_transform(data_reshaped, var, cfg.sqrt_transform)

    return data_reshaped


# -------------- NORMALISATION WITH UNIFORM PER MODEL -----------------------------------

def unnormalise_unif_per_model(data_norm, mode = "hr", data_type = "pr", sqrt_transform = False, 
                norm_method = "uniform_per_model", norm_stats=None, 
                root=None, logit=True,
                rcm = None, gcm = None):

    if data_type in ["pr", "sfcWind"] and sqrt_transform:
        name_str = "_sqrt"
    else:
        name_str = ""
    if logit:
        data_norm = torch.sigmoid(data_norm)
    if mode == "hr":
        s1 = 128
        s2 = 128
    elif mode == "lr":
        s1 = 20
        s2 = 36
    elif mode == "hr_avg":
        s1 = 8
        s2 = 8
    elif mode == "hr_avg_2":
        s1 = 32
        s2 = 32
        
    if norm_method == "uniform_per_model":
        if norm_stats is None and mode == "hr":    
            ns_path = os.path.join(root, "norm_stats", "hr_norm_stats_ecdf_matrix_" + data_type + "_train_" + rcm + "_" + gcm + "_FULL-PERIOD_NOISY" + name_str + ".pt")
            norm_stats = torch.load(ns_path)
        elif norm_stats is None and mode == "hr_avg":
            ns_path = os.path.join(root, "norm_stats", "hr_avg8x8_norm_stats_ecdf_matrix_" + data_type + "_train_" + "SUBSAMPLE" + name_str + ".pt")
            norm_stats = torch.load(ns_path)      
        
        len_full_data = norm_stats.shape[0]
        probs = torch.linspace(1, len_full_data, len_full_data)  / (len_full_data + 1)    
        data = torch.zeros(data_norm.shape[0], s1, s2)
        data_norm = data_norm.view(data_norm.shape[0], s1, s2)
    
        for i in range(s1):
            for j in range(s2):
                quantiles = norm_stats[:, i, j]
                data[:, i, j] = torch.tensor(np.interp(data_norm[:, i, j].detach().cpu().numpy(), probs, quantiles.detach().cpu().numpy()))
            
    else:
        data = data_norm.view(data_norm.shape[0], s1, s2)

    return data


def unnormalise_unif_per_model_batch(y, x, norm_method = "uniform_per_model",
                                     root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear",
                                     mode="hr", data_type="pr", sqrt_transform=False, logit=True,
                                     norm_stats_dict = None):
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    gcm_indices = x[:, -7:].nonzero(as_tuple=True)[1][::2].detach().cpu().numpy()
    rcm_indices = (x[:, -7:].nonzero(as_tuple=True)[1][1::2] - 3 ).detach().cpu().numpy()
    
    rcm_gcms = [(rcm_dict[rcm_list[i]], gcm_dict[gcm_list[i]]) for i in range(len(rcm_list))]

    list_run_indices = []   
    for pair in zip(rcm_indices, gcm_indices):
        for i in range(len(rcm_gcms)):
            if rcm_gcms[i] == pair:
                list_run_indices.append(i)
                
    splits = np.array(list_run_indices) # get the indices of the non-zero elements in one_hot

    # split the batch according to the values in one_hot
    unique_splits = np.unique(splits)
    n_unique_splits = len(unique_splits)
    batches = [y[splits == unique_splits[i]] for i in range(n_unique_splits)]
    
    # pass each split to the corresponding model and keep track of the original indices
    outputs = [((splits == unique_splits[i]).nonzero()[0], 
                unnormalise_unif_per_model(batches[i],
                            mode=mode, data_type=data_type, sqrt_transform=sqrt_transform, 
                            norm_method=norm_method, root=root,
                            rcm = rcm_list[i], gcm = gcm_list[i], logit = logit,
                            norm_stats=norm_stats_dict[(rcm_list[i], gcm_list[i])]
                            )
                
                ) for i in range(n_unique_splits) if batches[i].size(0) > 0]
    
    # Create an empty tensor to hold the reordered outputs
    reordered_outputs = torch.empty((y.shape[0], 128, 128))

    # Place each processed batch back into the original positions
    for indices, output in outputs:
        reordered_outputs[indices] = output
        
    return reordered_outputs

# --------- MULTIVARIATE NORMALISATION HELPERS -----------------------------------

def unnormalise_multivariate(data_norm, mode = "hr", data_types = ["tas"], sqrt_transform = True, 
                norm_method = "primitive", norm_stats=None, 
                root=None, final_square=True,
                sep_mean_std=False,
                n_keep_vals = 1000, verbose=False, len_full_data=int(3e4), logit=False,
                x=None):
    
    data_vars = []
    for i in range(len(data_types)):
        data_type = data_types[i]
        data_norm_var = data_norm[:, i, :]
        data_vars.append(unnormalise(data_norm_var, mode = mode, data_type = data_type,
                    sqrt_transform = sqrt_transform, 
                    norm_method = norm_method, norm_stats = norm_stats[data_type],
                    root = root, final_square = final_square, 
                    sep_mean_std = sep_mean_std, n_keep_vals = n_keep_vals, verbose = verbose,
                    len_full_data=len_full_data,
                    logit = logit, x=x))
        
    return torch.stack(data_vars, dim=1)        


# -------------- FFT HELPERS ---------------------------------------------------    
    
def fftpred2y(gen):
    ft_real = gen[:, :gen.size(1)//2].view(gen.size(0), 128,128)
    ft_imag = gen[:, gen.size(1)//2:].view(gen.size(0), 128,128)
    ft = torch.complex(ft_real, ft_imag)
    ift = torch.fft.ifft2(ft)
    ift_flattened = ift.reshape(ift.size(0), -1)
    return ift_flattened.real

# ------------- save loss curve -----------------------------------------

def losses_to_img(save_dir, file = "log.txt", mode = "", suffix = "", min_epoch = 5):
        
    train_epochs = []
    train_losses = []
    test_epochs = []
    test_losses = []

    with open(save_dir + file, 'r') as f:
        for line in f:
            if len(mode) > 0:
                train_match = re.match(r'Train-{}\s+\[Epoch (\d+)\]\s+loss: ([\d\.]+), s1: [\d\.]+, s2: [\d\.]+'.format(mode), line)
                test_match = re.match(r'Test-{}\s+\[Epoch (\d+)\]\s+loss: ([\d\.]+), s1: [\d\.]+, s2: [\d\.]+'.format(mode), line)
            else:
                train_match = re.match(r'Train\s+\[Epoch (\d+)\]\s+loss: ([\d\.]+), s1: [\d\.]+, s2: [\d\.]+', line)
                test_match = re.match(r'Test\s+\[Epoch (\d+)\]\s+loss: ([\d\.]+), s1: [\d\.]+, s2: [\d\.]+', line)
            
            if train_match:
                epoch, loss = train_match.groups()
                if int(epoch) > min_epoch:  # Only append if epoch is larger than 5
                    train_epochs.append(int(epoch))
                    train_losses.append(float(loss))
            elif test_match:
                epoch, loss = test_match.groups()
                if int(epoch) > min_epoch:  # Only append if epoch is larger than 5
                    test_epochs.append(int(epoch))
                    test_losses.append(float(loss))    

    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_losses, label='Train Loss')
    plt.plot(test_epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_dir + f"/losses{suffix}.png")
    plt.close()
    
# ------------- SELECT QUANTILES -------------------------------------------------

def extract_random_patch(x, patch_size = 8, top = None, left = None):
    assert len(x.shape) == 2  
    img_size = int(np.sqrt(x.shape[1]))
    x = x.view(-1, img_size, img_size)
    x = F.pad(x, (patch_size, patch_size, patch_size, patch_size))
    if top is None:
        top = np.random.randint(0, img_size + patch_size)
    if left is None:
        left = np.random.randint(0, img_size + patch_size)
    x = x[:, top:top+patch_size, left:left+patch_size].reshape(-1, patch_size**2)
    return x
    
# ------------ BATCH-WISE QUANTILE COMPUTATION -------------------------------------
def batchwise_quantile_clipping(data, q, batch_size = 100):
    quantiles_lower = []
    quantiles_upper = []

    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size]
        quantiles_lower.append(torch.quantile(batch, q).item())
        quantiles_upper.append(torch.quantile(batch, 1 - q).item())

    # Use the median of the computed quantiles as an approximation
    lower_bound = torch.tensor(quantiles_lower).median().item()
    upper_bound = torch.tensor(quantiles_upper).median().item()

    data = torch.clamp(data, lower_bound, upper_bound)
    return data
    
# ------------ EVALUATION -------------------------------------------------------------------------------

from scipy.stats import rankdata
def compute_rank_histogram(ground_truth, forecasts, axis=0, method="min"):
    # assume that forecasts have an extra dimension 0 for the several samples
    # other dimensions should be equal to ground truth
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(forecasts, torch.Tensor):
        forecasts = forecasts.detach().cpu().numpy()
    combined=np.concatenate((np.expand_dims(ground_truth, axis = axis), forecasts), axis = axis)
    ranks=np.apply_along_axis(lambda x: rankdata(x,method=method),axis,combined)
    if axis == 0:
        ranks_truth = ranks[0, :]
    elif axis == 1 or axis == -1:
        ranks_truth = ranks[:, 0]
    else:
        raise ValueError("axis must be 0, 1 or -1")
    result = np.histogram(ranks_truth, bins=np.linspace(0.5, combined.shape[axis]+0.5, combined.shape[axis]+1))    
    return result, np.mean(ranks_truth), np.var(ranks_truth)


# ------------ MAKE DATALOADER --------------------------------------------------------------------------


def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0):
    """Make dataloader.

    Args:
        x (torch.Tensor): data of predictors.
        y (torch.Tensor): data of responses.
        batch_size (int, optional): batch size. Defaults to 128.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.

    Returns:
        DataLoader: data loader
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def gaussian_bump(H, W, center, sigma):
    y = torch.arange(H).float()
    x = torch.arange(W).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2*sigma**2))

def add_gaussian_bump(xc, center, sigma, size, n_vars=4, amplitude=1.0, scale_std=1.0):
    """
    Adds a fixed bump shape, but scales it with a random Gaussian factor per (sample, variable).
    
    scale_std: std deviation of scaling factors (mean is 0)
    """
    batch = xc.shape[0]
    H, W = size, size
    xc_2d = xc.view(batch, n_vars, H, W)

    # Base bump (fixed shape)
    bump = gaussian_bump(H, W, center, sigma).to(xc_2d.device)  # (H, W)

    # Random scaling factors for each (sample, variable)
    scales = torch.randn(batch, n_vars, device=xc_2d.device) * scale_std  # (batch, n_vars)

    # Expand bump to (batch, n_vars, H, W) with scaling
    bumps_scaled = scales[:, :, None, None] * bump  # broadcast multiply

    # Add to data
    perturbed_2d = xc_2d + amplitude * bumps_scaled
    return perturbed_2d.view(batch, -1)
