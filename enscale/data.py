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

    def lr_path(self, var, run: RunSpec, folder: str = "train", file_suffix: str = ""):
        return self.lr_pattern.format(
            root=self.root,
            folder=folder,
            var=var,
            gcm=run.gcm,
            rcm=run.rcm,
            variant=run.variant,
            file_suffix=file_suffix,
        )

    def hr_path(self, var, run: RunSpec, folder: str = "train", file_suffix: str = ""):
        return self.hr_pattern.format(
            root=self.root,
            folder=folder,
            var=var,
            gcm=run.gcm,
            rcm=run.rcm,
            variant=run.variant,
            file_suffix=file_suffix,
        )
        

class CordexReader(DataReader):
    def __init__(self, path_spec: CordexPathSpec, modes: dict = None, default_mode: str = "train"):
        self.path_spec = path_spec
        self.modes = modes or {}
        self.default_mode = default_mode

    def _mode_info(self, mode: str):
        if mode is None:
            mode = self.default_mode
        return self.modes.get(mode, {"folder": "train", "file_suffix": ""})

    def load_lr(self, var, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.lr_path(var, run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        return xr.open_dataset(path)[var]

    def load_hr(self, var, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.hr_path(var, run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        return xr.open_dataset(path)[var]
    
    def load_time(self, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.hr_path(var="tas", run=run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        ds = xr.open_dataset(path)
        return ds["time"]


class SimpleNetCDFReader(DataReader):
    def __init__(self, files):
        self.files = files  # dict(var -> path)

    def load_lr(self, var, run):
        return xr.open_dataset(self.files["lr"][var])[var]

    def load_hr(self, var, run):
        return xr.open_dataset(self.files["hr"][var])[var]


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
        kernel_size_lr,
        kernel_size_hr,
        include_time=True,
        include_year=False,
        mode: str = "train",
        return_timepair: bool = False,
    ):
        self.reader = reader
        self.preproc = preproc
        self.run = run
        self.mode = mode

        # Load and preprocess all data ONCE
        lr_vars = []
        for v in variables_lr:
            # pass the requested mode to the reader (for cordex reader this selects folder/file_suffix)
            x = reader.load_lr(v, run, mode=self.mode)
            x = torch.from_numpy(x.values).float()
            x = preproc.process_lr(x, v)
            lr_vars.append(x)

        hr_vars = []
        hr_coarse_vars = []
        for v in variables_hr:
            y = reader.load_hr(v, run, mode=self.mode)
            y = torch.from_numpy(y.values).float()
            ny, nx = y.shape[-2], y.shape[-1]
            y = preproc.process_hr(y, v)
            hr_coarse_vars.append(
                preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_lr, kernel_size_lr, 0).view(y.shape[0], 1, -1)
            )
            if kernel_size_hr > 1:
                y = preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_hr, kernel_size_hr, 0).view(y.shape[0], -1)
            hr_vars.append(y.unsqueeze(1))

        self.x = torch.cat(lr_vars, dim=1)
        self.z = torch.cat(hr_coarse_vars, dim=1)
        self.y = torch.cat(hr_vars, dim=1)

        self.include_time = include_time
        self.include_year = include_year
        self.return_timepair = return_timepair

        # Optional time features
        if self.include_time or self.include_year:
            time_index = reader.load_time(run, mode=self.mode)

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
        if self.return_timepair:
            return self.x.shape[0] - 1
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.return_timepair:
            # Temporal contract: previous timestep first, then current timestep.
            return (
                self.x[idx],
                self.z[idx],
                self.y[idx],
                self.x[idx + 1],
                self.z[idx + 1],
                self.y[idx + 1],
            )
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
        modes = getattr(data.cordex, 'modes', None)
        return CordexReader(spec, modes=modes, default_mode="train")

    elif data.type == "single_pair":
        return SimpleNetCDFReader(data.inputs)

def build_one_hot(cfg, gcm_list, rcm_list):
    # support cfg.data as dataclass or dict
    if isinstance(cfg.data, dict):
        enc_cfg = cfg.data.get("ensemble_encoding", {})
    else:
        enc_cfg = getattr(cfg.data, 'ensemble_encoding', {}) or {}

    if not isinstance(enc_cfg, dict) or not enc_cfg.get("enabled", False):
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
            get_rcm_gcm_combinations(data_cfg.cordex.root)

        # Prefer legacy flat fields when provided; otherwise use nested runs config.
        legacy_run_indices = getattr(data_cfg, "run_indices", None)
        legacy_n_models = getattr(data_cfg, "n_models", None)

        if legacy_run_indices is not None:
            indices = [int(i) for i in legacy_run_indices]
        elif legacy_n_models is not None:
            indices = list(range(int(legacy_n_models)))
        elif data_cfg.runs.selection == "first_n":
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


def get_data(cfg,
             test_size: float = 0.0,
             shuffle: bool = True,
             mode: str = "train",
             batch_size = None,
             temporal: Optional[bool] = None):
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

    if temporal is None:
        temporal = bool(getattr(cfg.data, "return_timepair", False))

    # 4. Datasets (one per run)
    datasets = [
        DownscalingDataset(
            reader=reader,
            preproc=preproc,
            run=run,
            variables_lr=cfg.data.variables_lr,
            variables_hr=cfg.data.variables,
            kernel_size_lr=cfg.data.kernel_size_lr,
            kernel_size_hr=cfg.data.kernel_size_hr,
            mode=mode,
            return_timepair=temporal,
        )
        for run in runs
    ]

    # 5. Concatenate
    full_dataset = ConcatDataset(datasets)
    
    if batch_size is None:
        batch_size = cfg.training.batch_size
    if test_size > 0:
        train_indices, test_indices = train_test_split(list(range(len(full_dataset))), 
                                                       test_size = test_size, 
                                                       random_state = cfg.data.random_state)
        dataset_train = Subset(full_dataset, train_indices)
        dataset_test = Subset(full_dataset, test_indices)
        #dataloader_train = DataLoader(dataset_train, batch_size, shuffle=shuffle)
        #dataloader_test = DataLoader(dataset_test, batch_size, shuffle=shuffle)
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
            )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
            )
    else:
        dataloader_train = DataLoader(full_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle)
        dataloader_test = None

    return dataloader_train, dataloader_test