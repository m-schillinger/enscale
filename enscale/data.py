from dataclasses import dataclass
import scipy
from utils import correct_units
from typing import Optional
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import ConcatDataset
import os
import numpy as np
import re
import xarray as xr
from utils import *
import scipy
import math

# what is missing in this file:
# option to specify path for Cordex data

@dataclass(frozen=True)
class RunSpec:
    """
    Describes ONE input–output pairing.
    Works for:
    - single-pair training
    - multi RCM/GCM ensembles
    """
    gcm: Optional[str] = None
    rcm: Optional[str] = None
    variant: str = "r1i1p1"

    one_hot: Optional[torch.Tensor] = None


class DataReader:
    def load_lr(self, var: str, run: RunSpec):
        raise NotImplementedError

    def load_hr(self, var: str, run: RunSpec):
        raise NotImplementedError

    def load_time(self, run: RunSpec):
        raise NotImplementedError

from dataclasses import dataclass

@dataclass(frozen=True)
class CordexPathSpec:
    root: str
    lr_pattern: str
    hr_pattern: str

    def lr_path(self, var, run: RunSpec):
        return self.lr_pattern.format(
            root=self.root,
            var=var,
            gcm=run.gcm,
            rcm=run.rcm,
            variant=run.variant,
        )

    def hr_path(self, var, run: RunSpec):
        return self.hr_pattern.format(
            root=self.root,
            var=var,
            gcm=run.gcm,
            rcm=run.rcm,
            variant=run.variant,
        )
        

class CordexReader(DataReader):
    def __init__(self, path_spec: CordexPathSpec):
        self.path_spec = path_spec

    def load_lr(self, var, run: RunSpec):
        path = self.path_spec.lr_path(var, run)
        return xr.open_dataset(path)[var]

    def load_hr(self, var, run: RunSpec):
        path = self.path_spec.hr_path(var, run)
        return xr.open_dataset(path)[var]
    
    def load_time(self, run: RunSpec):
        # load time from one of the variables
        path = self.path_spec.hr_path(var="tas", run=run)
        ds = xr.open_dataset(path)
        return ds["time"]


class SimpleNetCDFReader(DataReader):
    def __init__(self, files):
        self.files = files  # dict(var -> path)

    def load_lr(self, var, run):
        return xr.open_dataset(self.files["lr"][var])[var]

    def load_hr(self, var, run):
        return xr.open_dataset(self.files["hr"][var])[var]

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

def ecdf_normalise(x, norm_stats, len_full_data=int(3e4)): # TO DO: need to make more flexible
    data_norm = torch.zeros_like(x)
    probs = torch.linspace(1, len_full_data, len_full_data) / (len_full_data + 1) 
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            quantiles = norm_stats[:, i, j]
            data_norm[:, i, j] = torch.tensor(np.interp(x[:, i, j].detach().cpu().numpy(), quantiles.detach().cpu().numpy(), probs))
    return data_norm

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

class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def process_lr(self, x, var):
        return normalise(
            x,
            var=var,
            mode="lr",
            cfg=self.cfg,
        )

    def process_hr(self, x, var):
        return normalise(
            x,
            var=var,
            mode="hr",
            cfg=self.cfg,
        )

    def coarsen(self, x, k, stride, padding):
        return torch.nn.functional.avg_pool2d(
            x.unsqueeze(1), kernel_size=k, stride=stride, padding=padding
        ).squeeze(1)


class DownscalingDataset(Dataset):
    """
    Unified dataset:
    - works for single-pair or multi-pair
    - no CORDEX logic
    - no server paths
    """

    def __init__(
        self,
        reader: DataReader,
        preproc: Preprocessor,
        run: RunSpec,
        variables_lr,
        variables_hr,
        kernel_size,
        include_time=True,
        include_year=False,
    ):
        self.reader = reader
        self.preproc = preproc
        self.run = run

        # Load and preprocess all data ONCE
        lr_vars = []
        for v in variables_lr:
            x = reader.load_lr(v, run)
            x = torch.from_numpy(x.values).float()
            x = preproc.process_lr(x, v)
            lr_vars.append(x)

        hr_vars = []
        hr_coarse_vars = []
        for v in variables_hr:
            y = reader.load_hr(v, run)
            y = torch.from_numpy(y.values).float()
            ny, nx = y.shape[-2], y.shape[-1]
            y = preproc.process_hr(y, v)
            hr_vars.append(y.unsqueeze(1))
            hr_coarse_vars.append(
                preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size, kernel_size, 0).view(y.shape[0], 1, -1)
            )

        self.x = torch.cat(lr_vars, dim=1)
        self.z = torch.cat(hr_coarse_vars, dim=1)
        self.y = torch.cat(hr_vars, dim=1)

        self.include_time = include_time
        self.include_year = include_year

        # Optional time features
        if self.include_time or self.include_year:
            time_index = reader.load_time(run)

            time_feats_1d = self._build_time_features(time_index)
            assert time_feats_1d is not None
            assert time_feats_1d.shape[0] == self.x.shape[0]

            #time_feats = self._expand_time_features(time_feats_1d, self.x)
            time_feats = time_feats_1d
            self.x = torch.cat([self.x, time_feats], dim=1)

        # Optional metadata
        if run.one_hot is not None:
            oh = run.one_hot.expand(self.x.shape[0], -1)
            self.x = torch.cat([self.x, oh], dim=-1)
      
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.z[idx], self.y[idx]

    def _build_time_features(self, time_index):
        """
        Builds time features exactly as in the original code.
        Returns tensor of shape (T, C_time)
        """
        idx = time_index.to_index()
        months = idx.strftime("%m").astype(int).to_numpy()
        days   = idx.strftime("%d").astype(int).to_numpy()
        years  = idx.strftime("%Y").astype(int).to_numpy()
        is_leap = is_leap_year(years)
        leap_year_mask = is_leap & (months == 2) & (days == 29)
        consider_leap = bool(np.any(leap_year_mask))

        doy = day_of_year_vectorized(
            months,
            days,
            is_leap,
            consider_leap=consider_leap,
        )

        doy = torch.from_numpy(doy).float().unsqueeze(1)
        year = torch.from_numpy(years).float().unsqueeze(1)

        feats = []

        if self.include_year:
            feats.append(year)

        if self.include_time:
            feats.extend([
                doy,
                torch.sin((365 / (2 * math.pi)) * doy),
                torch.cos((365 / (2 * math.pi)) * doy),
                torch.sin((365 / math.pi) * doy),
                torch.cos((365 / math.pi) * doy),
            ])

        if not feats:
            return None

        return torch.cat(feats, dim=1)  # (T, C_time)


    
def build_reader(cfg):
    data = cfg.data

    if data.type == "cordex_ensemble":
        spec = CordexPathSpec(
            root=data.cordex.root,
            lr_pattern=data.cordex.lr_pattern,
            hr_pattern=data.cordex.hr_pattern,
        )
        return CordexReader(spec)

    elif data.type == "single_pair":
        return SimpleNetCDFReader(data.inputs)

def build_one_hot(cfg, gcm_list, rcm_list):
    enc_cfg = cfg.data.get("ensemble_encoding", {})
    if not enc_cfg.get("enabled", False):
        return None

    scheme = enc_cfg.get("scheme", "gcm+rcm")

    if scheme != "gcm+rcm":
        raise NotImplementedError(f"Unknown encoding scheme: {scheme}")

    gcms = sorted(set(gcm_list))
    rcms = sorted(set(rcm_list))

    gcm_index = {g: i for i, g in enumerate(gcms)}
    rcm_index = {r: i for i, r in enumerate(rcms)}

    n_runs = len(gcm_list)
    n_feat = len(gcms) + len(rcms)

    one_hot = torch.zeros(n_runs, n_feat)

    for i, (gcm, rcm) in enumerate(zip(gcm_list, rcm_list)):
        one_hot[i, gcm_index[gcm]] = 1.0
        one_hot[i, len(gcms) + rcm_index[rcm]] = 1.0

    return one_hot


def build_run_specs(cfg):
    data_cfg = cfg.data

    # ---------- Single pair ----------
    if data_cfg.type == "single_pair":
        return [RunSpec()]  # exactly one run, no metadata

    # ---------- Multi RCM/GCM ----------
    elif data_cfg.type == "cordex_ensemble":
        gcm_list, rcm_list, gcm_dict, rcm_dict = \
            get_rcm_gcm_combinations(data_cfg.root)

        if data_cfg.runs.selection == "first_n":
            indices = list(range(data_cfg.runs.n_models))
        elif data_cfg.runs.selection == "explicit":
            indices = data_cfg.runs.indices
        else:
            raise ValueError("Unknown run selection")

        one_hot = build_one_hot(cfg, gcm_list, rcm_list)

        runs = []
        for i in indices:
            runs.append(
                RunSpec(
                    gcm=gcm_list[i],
                    rcm=rcm_list[i],
                    one_hot=one_hot[i] if one_hot is not None else None,
                )
            )
        return runs


def get_data(cfg):
    """
    Builds dataloaders from config.
    Works for:
    - single input–output pair
    - multi RCM/GCM ensemble
    """

    # 1. Reader (I/O only)
    reader = build_reader(cfg)
    
    # 2. Preprocessor (math only)
    preproc = Preprocessor(cfg.data.preprocessing)

    # 3. Run specifications
    runs = build_run_specs(cfg)

    # 4. Datasets (one per run)
    datasets = [
        DownscalingDataset(
            reader=reader,
            preproc=preproc,
            run=run,
            variables_lr=cfg.data.variables_lr,
            variables_hr=cfg.data.variables,
            kernel_size=cfg.data.kernel_size_lr,
        )
        for run in runs
    ]

    # 5. Concatenate
    full_dataset = ConcatDataset(datasets)

    # 6. DataLoader
    loader = DataLoader(
        full_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    return loader