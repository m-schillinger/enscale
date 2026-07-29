from dataclasses import dataclass
import scipy
from utils import correct_units
from typing import Optional
import os
import glob
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
    def load_lr(self, var: str, run: RunSpec, mode: str = None):
        raise NotImplementedError

    def load_hr(self, var: str, run: RunSpec, mode: str = None):
        raise NotImplementedError

    def load_hr_normalized(self, var: str, run: RunSpec, mode: str = None):
        return self.load_hr(var, run, mode=mode)

    def load_time(self, run: RunSpec, mode: str = None):
        raise NotImplementedError

from dataclasses import dataclass

@dataclass(frozen=True)
class CordexPathSpec:
    root: str
    lr_pattern: str
    hr_pattern: str
    normalized_hr_pattern: str = ""
    normalized_hr_root: str = ""

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

    def normalized_hr_path(self, var, run: RunSpec, folder: str = "train", file_suffix: str = "", norm_suffix: str = ""):
        if not self.normalized_hr_pattern:
            raise ValueError("normalized_hr_pattern is not configured")
        root = self.normalized_hr_root or self.root
        return self.normalized_hr_pattern.format(
            root=root,
            folder=folder,
            var=var,
            gcm=run.gcm,
            rcm=run.rcm,
            variant=run.variant,
            file_suffix=file_suffix,
            norm_suffix=norm_suffix,
        )
        

class CordexReader(DataReader):
    def __init__(self, path_spec: CordexPathSpec, modes: dict = None, default_mode: str = "train", preproc_cfg=None, precip_zeros: str = "random"):
        self.path_spec = path_spec
        self.modes = modes or {}
        self.default_mode = default_mode
        self.preproc_cfg = preproc_cfg
        self.precip_zeros = precip_zeros

    def _mode_info(self, mode: str):
        if mode is None:
            mode = self.default_mode
        if mode in self.modes:
            return self.modes[mode]
        if mode == "valid":
            return self.modes.get("validation", self.modes.get("val", {"folder": "valid", "file_suffix": ""}))
        if mode == "test":
            return self.modes.get("test", {"folder": "test", "file_suffix": ""})
        if mode == "inference":
            return self.modes.get("inference", {"folder": "train", "file_suffix": ""})
        if mode in {"test_interpolation", "test_extrapolation"}:
            return self.modes.get(mode, {"folder": "train", "file_suffix": ""})
        return {"folder": "train", "file_suffix": ""}

    def _normalized_mode_info(self, mode: str):
        cfg = self.preproc_cfg
        mode_key = self.default_mode if mode is None else mode
        norm_modes = getattr(cfg, "normalized_hr_modes", {}) if cfg is not None else {}

        if mode_key in norm_modes:
            return norm_modes[mode_key]
        if mode_key == "valid":
            return norm_modes.get("validation", norm_modes.get("val", self._mode_info(mode_key)))
        if mode_key == "test":
            return norm_modes.get("test", self._mode_info(mode_key))
        if mode_key == "inference":
            return norm_modes.get("inference", self._mode_info(mode_key))

        # Legacy fallback for precomputed uniform files when folders differ from raw data.
        norm_method_out = str(getattr(cfg, "norm_method_output", "") or "")
        if norm_method_out in {"uniform", "uniform_per_model"}:
            legacy_uniform_modes = {
                "train": {"folder": "train_norm_unif", "file_suffix": "_train-period"},
                "test_interpolation": {"folder": "test_norm_unif/interpolation", "file_suffix": "_2030-2039"},
                "test_extrapolation": {"folder": "test_norm_unif/extrapolation", "file_suffix": "_2090-2099"},
            }
            if mode_key in legacy_uniform_modes:
                return legacy_uniform_modes[mode_key]

        return self._mode_info(mode_key)

    def load_lr(self, var, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.lr_path(var, run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        return xr.open_dataset(path)[var]

    def load_hr(self, var, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.hr_path(var, run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        return xr.open_dataset(path)[var]

    def _normalized_hr_suffix(self, var: str) -> str:
        cfg = self.preproc_cfg
        if cfg is None:
            return ""

        suffix_map = getattr(cfg, "normalized_hr_suffix_by_var", {}) or {}
        suffix_template = suffix_map.get(var, getattr(cfg, "normalized_hr_default_suffix", ""))
        if suffix_template is None:
            return ""

        suffix = str(suffix_template).format(
            var=var,
            precip_zeros=self.precip_zeros,
            filter_tag="filtered" if getattr(cfg, "normalized_hr_filter_outliers", False) else "",
        )
        return suffix

    def load_hr_normalized(self, var, run: RunSpec, mode: str = None):
        mi = self._normalized_mode_info(mode)
        path = self.path_spec.normalized_hr_path(
            var,
            run,
            folder=mi.get("folder", "train"),
            file_suffix=mi.get("file_suffix", ""),
            norm_suffix=self._normalized_hr_suffix(var),
        )
        return xr.open_dataset(path)[var]
    
    def load_time(self, run: RunSpec, mode: str = None):
        mi = self._mode_info(mode)
        path = self.path_spec.hr_path(var="tas", run=run, folder=mi.get("folder", "train"), file_suffix=mi.get("file_suffix", ""))
        ds = xr.open_dataset(path)
        return ds["time"]


class PatternReader(DataReader):
    def __init__(self, patterns: dict, modes: dict = None, default_mode: str = "train"):
        self.patterns = patterns or {}
        self.modes = modes or {}
        self.default_mode = default_mode

    def _mode_config(self, mode: str):
        if mode is None:
            mode = self.default_mode
        if mode in self.modes:
            return self.modes[mode]
        if mode == "valid":
            return self.modes.get("validation", self.modes.get("val", {}))
        if mode == "test":
            return self.modes.get("test", {})
        if mode == "inference":
            return self.modes.get("inference", {})
        return {}

    def _resolve_path(self, var: str, run: RunSpec, kind: str, mode: str = None):
        mode_cfg = self._mode_config(mode)
        base_pattern = mode_cfg.get(f"{kind}_pattern", self.patterns.get(f"{kind}_pattern"))
        if not base_pattern:
            raise ValueError(f"No {kind} pattern configured for mode='{mode}'")
        if isinstance(base_pattern, (list, tuple)):
            matches = []
            for pattern in base_pattern:
                matches.extend(glob.glob(str(pattern).format(var=var, gcm=run.gcm or "", rcm=run.rcm or "", variant=run.variant)))
            matches = sorted(set(matches))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {kind} file for {var}, got {len(matches)}: {matches}")
            return matches[0]
        return str(base_pattern).format(var=var, gcm=run.gcm or "", rcm=run.rcm or "", variant=run.variant)

    def load_lr(self, var, run: RunSpec, mode: str = None):
        return xr.open_dataset(self._resolve_path(var, run, "lr", mode=mode))[var]

    def load_hr(self, var, run: RunSpec, mode: str = None):
        return xr.open_dataset(self._resolve_path(var, run, "hr", mode=mode))[var]

    def load_time(self, run: RunSpec, mode: str = None):
        path = self._resolve_path("tas", run, "hr", mode=mode)
        return xr.open_dataset(path)["time"]


class SimpleNetCDFReader(DataReader):
    def __init__(self, files):
        self.files = files  # dict(var -> path)

    def load_lr(self, var, run, mode: str = None):
        return xr.open_dataset(self.files["lr"][var])[var]

    def load_hr(self, var, run, mode: str = None):
        return xr.open_dataset(self.files["hr"][var])[var]

    def load_time(self, run: RunSpec, mode: str = None):
        hr_vars = self.files.get("hr", {})
        if not hr_vars:
            raise ValueError("No HR files configured to infer time index")
        first_var = next(iter(hr_vars.keys()))
        return xr.open_dataset(hr_vars[first_var])["time"]


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

    def process_hr(self, x, var, skip_normalization: bool = False):
        return normalise(
            x,
            var=var,
            mode="hr",
            cfg=self.cfg,
            skip_normalization=skip_normalization,
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
        inference_mode: bool = False,
        inference_fill_value: str = "zeros",
    ):
        self.reader = reader
        self.preproc = preproc
        self.run = run
        self.mode = mode
        self.use_precomputed_hr = bool(getattr(self.preproc.cfg, "use_precomputed_hr", False))

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
        if not inference_mode:
            for v in variables_hr:
                if self.use_precomputed_hr:
                    y = reader.load_hr_normalized(v, run, mode=self.mode)
                else:
                    y = reader.load_hr(v, run, mode=self.mode)
                y = torch.from_numpy(y.values).float()
                if y.ndim != 3:
                    raise ValueError(f"Expected HR tensor with shape (time, y, x), got shape {tuple(y.shape)} for var '{v}'")
                ny, nx = y.shape[-2], y.shape[-1]
                y = preproc.process_hr(y, v, skip_normalization=self.use_precomputed_hr)
                hr_coarse_vars.append(
                    preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_lr, kernel_size_lr, 0).view(y.shape[0], 1, -1)
                )
                if kernel_size_hr > 1:
                    y = preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_hr, kernel_size_hr, 0).view(y.shape[0], -1)
                hr_vars.append(y.unsqueeze(1))

        self.x = torch.cat(lr_vars, dim=1)
        if inference_mode:
            self.z = self._make_target_placeholder(self.x.shape[0], max(1, self.x.shape[-1]), fill_value=inference_fill_value)
            self.y = self._make_target_placeholder(self.x.shape[0], max(1, self.x.shape[-1]), fill_value=inference_fill_value)
        else:
            self.z = torch.cat(hr_coarse_vars, dim=1)
            self.y = torch.cat(hr_vars, dim=1)

        self.include_time = include_time
        self.include_year = include_year
        self.return_timepair = return_timepair
        self.inference_mode = inference_mode
        self.inference_fill_value = inference_fill_value

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

    def _make_target_placeholder(self, n_samples, n_features, fill_value="zeros"):
        if fill_value == "nan":
            return torch.full((n_samples, 1, n_features), float("nan"), dtype=torch.float32)
        return torch.zeros((n_samples, 1, n_features), dtype=torch.float32)

    def _build_time_features(self, time_index):
        """
        Builds time features exactly as in the original code.
        Returns tensor of shape (T, C_time)
        """
        if hasattr(time_index, "to_index"):
            idx = time_index.to_index()
        else:
            idx = time_index

        if hasattr(idx, "strftime"):
            months = idx.strftime("%m").astype(int).to_numpy()
            days = idx.strftime("%d").astype(int).to_numpy()
            years = idx.strftime("%Y").astype(int).to_numpy()
        else:
            months = []
            days = []
            years = []
            for v in idx:
                if hasattr(v, "month") and hasattr(v, "day") and hasattr(v, "year"):
                    months.append(int(v.month))
                    days.append(int(v.day))
                    years.append(int(v.year))
                else:
                    months.append(1)
                    days.append(1)
                    years.append(2000)
            months = np.array(months, dtype=int)
            days = np.array(days, dtype=int)
            years = np.array(years, dtype=int)

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
            normalized_hr_pattern=getattr(data.preprocessing, "normalized_hr_pattern", ""),
            normalized_hr_root=getattr(data.preprocessing, "normalized_hr_root", ""),
        )
        modes = getattr(data.cordex, 'modes', None)
        return CordexReader(
            spec,
            modes=modes,
            default_mode="train",
            preproc_cfg=data.preprocessing,
            precip_zeros=getattr(data, "precip_zeros", "random"),
        )

    elif data.type == "single_pair":
        return SimpleNetCDFReader(data.inputs)

    elif data.type == "pattern":
        return PatternReader(
            patterns=getattr(data, "patterns", {}) or {},
            modes=getattr(data, "modes", None),
            default_mode="train",
        )


def get_ensemble_encoding_scheme(cfg):
    # Support cfg.data as dataclass or dict.
    if isinstance(cfg.data, dict):
        enc_cfg = cfg.data.get("ensemble_encoding", {})
    else:
        enc_cfg = getattr(cfg.data, "ensemble_encoding", {}) or {}

    if not isinstance(enc_cfg, dict) or not enc_cfg.get("enabled", False):
        return None

    scheme = str(enc_cfg.get("scheme", "gcm+rcm"))
    valid_schemes = {"gcm", "rcm", "gcm+rcm"}
    if scheme not in valid_schemes:
        raise ValueError(
            f"Unknown ensemble encoding scheme: {scheme}. "
            f"Expected one of {sorted(valid_schemes)}"
        )
    return scheme

def build_one_hot(cfg, gcm_list, rcm_list):
    scheme = get_ensemble_encoding_scheme(cfg)
    if scheme is None:
        return None

    gcms = sorted(set(gcm_list))
    rcms = sorted(set(rcm_list))

    n_runs = len(gcm_list)
    if scheme == "gcm":
        gcm_index = {g: i for i, g in enumerate(gcms)}
        one_hot = torch.zeros(n_runs, len(gcms))
        for i, gcm in enumerate(gcm_list):
            one_hot[i, gcm_index[gcm]] = 1.0
        return one_hot

    if scheme == "rcm":
        rcm_index = {r: i for i, r in enumerate(rcms)}
        one_hot = torch.zeros(n_runs, len(rcms))
        for i, rcm in enumerate(rcm_list):
            one_hot[i, rcm_index[rcm]] = 1.0
        return one_hot

    gcm_index = {g: i for i, g in enumerate(gcms)}
    rcm_index = {r: i for i, r in enumerate(rcms)}
    one_hot = torch.zeros(n_runs, len(gcms) + len(rcms))
    for i, (gcm, rcm) in enumerate(zip(gcm_list, rcm_list)):
        one_hot[i, gcm_index[gcm]] = 1.0
        one_hot[i, len(gcms) + rcm_index[rcm]] = 1.0

    return one_hot


def build_run_specs(cfg):
    data_cfg = cfg.data

    # ---------- Single pair / pattern-based ----------
    if data_cfg.type in {"single_pair", "pattern"}:
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
             temporal: Optional[bool] = None,
             validation_size: Optional[float] = None,
             validation_mode: Optional[str] = None,
             test_mode: Optional[str] = None,
             inference_mode: bool = False):
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

    if validation_mode is None:
        validation_mode = getattr(cfg.data, "validation_mode", None)
    if validation_size is None:
        validation_size = getattr(cfg.data, "validation_size", None)
    if test_mode is None:
        test_mode = getattr(cfg.data, "test_mode", None)

    inference_mode = inference_mode or bool(getattr(cfg.data, "inference_mode", False))
    validation_source = str(getattr(cfg.data, "validation_source", "auto")).lower()

    if batch_size is None:
        batch_size = cfg.training.batch_size

    def _make_dataset(dataset_mode: str):
        return ConcatDataset([
            DownscalingDataset(
                reader=reader,
                preproc=preproc,
                run=run,
                variables_lr=cfg.data.variables_lr,
                variables_hr=cfg.data.variables,
                kernel_size_lr=cfg.data.kernel_size_lr,
                kernel_size_hr=cfg.data.kernel_size_hr,
                mode=dataset_mode,
                return_timepair=temporal,
                inference_mode=inference_mode,
                inference_fill_value=getattr(cfg.data, "inference_fill_value", "zeros"),
            )
            for run in runs
        ])

    train_dataset = _make_dataset(mode)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
    dataloader_val = None

    if validation_mode and validation_source in {"folder", "auto"}:
        try:
            val_dataset = _make_dataset(validation_mode)
            dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
        except (FileNotFoundError, OSError, ValueError) as exc:
            if validation_source == "auto" and validation_size is not None and validation_size > 0:
                train_indices, val_indices = train_test_split(
                    list(range(len(train_dataset))),
                    test_size=validation_size,
                    random_state=cfg.data.random_state,
                )
                dataset_train = Subset(train_dataset, train_indices)
                dataset_val = Subset(train_dataset, val_indices)
                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
                dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
            elif validation_source == "auto":
                dataloader_val = None
            else:
                raise
    elif validation_size is not None and validation_size > 0:
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))),
            test_size=validation_size,
            random_state=cfg.data.random_state,
        )
        dataset_train = Subset(train_dataset, train_indices)
        dataset_val = Subset(train_dataset, val_indices)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)

    if test_mode:
        try:
            test_dataset = _make_dataset(test_mode)
            dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
        except (FileNotFoundError, OSError, ValueError):
            dataloader_test = None
    elif test_size > 0:
        test_indices = list(range(len(train_dataset)))
        dataset_test = Subset(train_dataset, test_indices)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers)
    else:
        dataloader_test = None

    if inference_mode:
        return dataloader_train, dataloader_val, dataloader_test

    if validation_mode and validation_source in {"folder", "auto"}:
        return dataloader_train, dataloader_val, dataloader_test
    if validation_size is not None and validation_size > 0:
        return dataloader_train, dataloader_val, dataloader_test

    return dataloader_train, dataloader_test