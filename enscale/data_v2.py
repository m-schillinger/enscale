from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import glob
import os

import math
import numpy as np
import torch
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from config_v2 import ConfigV2, DataConfigV2, ModeConfigV2
from utils import correct_units, day_of_year_vectorized, get_rcm_gcm_combinations, is_leap_year, normalise


@dataclass(frozen=True)
class RunSpecV2:
    gcm: Optional[str] = None
    rcm: Optional[str] = None
    variant: str = "r1i1p1"
    one_hot: Optional[torch.Tensor] = None


class DataReaderV2:
    def load_lr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        raise NotImplementedError

    def load_hr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        raise NotImplementedError

    def load_time(self, run: RunSpecV2, split: str, submode: Optional[str] = None):
        raise NotImplementedError


def _append_suffix(path: str, suffix: str) -> str:
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def _format_with_optional_var_suffix(template: str, context: Dict[str, str], var_suffix: str) -> str:
    if "{var_suffix}" in template:
        return template.format(var_suffix=var_suffix, **context)
    return _append_suffix(template.format(**context), var_suffix)


class _ModeAwarePathResolver:
    def __init__(self, cfg: ConfigV2):
        self.cfg = cfg
        self.cfg_data = cfg.data

    def mode(self, split: str, submode: Optional[str]) -> ModeConfigV2:
        if split in {"train", "validation", "valid", "val"}:
            return self.cfg_data.resolve_mode(split=split, submode=submode)
        if split in {"test", "inference"}:
            return self.cfg.inference.data_modes.resolve_mode(split=split, submode=submode)
        raise ValueError(f"Unsupported split '{split}'")

    def var_suffix(self, var: str, kind: str) -> str:
        return self.cfg_data.var_suffix(var, kind)


class CordexReaderV2(DataReaderV2):
    def __init__(self, cfg: ConfigV2):
        self.cfg = cfg
        self.cfg_data = cfg.data
        self.resolver = _ModeAwarePathResolver(cfg)

    def _template(self, kind: str, mode_cfg: ModeConfigV2) -> str:
        if kind == "lr":
            return mode_cfg.lr_pattern or self.cfg_data.cordex.lr_pattern
        if kind == "hr":
            return mode_cfg.hr_pattern or self.cfg_data.cordex.hr_pattern
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def _folder_for_kind(mode_cfg: ModeConfigV2, kind: str) -> str:
        if kind == "lr":
            return mode_cfg.folder_lr if mode_cfg.folder_lr is not None else mode_cfg.folder
        if kind == "hr":
            return mode_cfg.folder_hr if mode_cfg.folder_hr is not None else mode_cfg.folder
        raise ValueError(f"Unknown kind: {kind}")

    def _path(self, var: str, run: RunSpecV2, kind: str, split: str, submode: Optional[str] = None) -> str:
        mode_cfg = self.resolver.mode(split, submode)
        template = self._template(kind, mode_cfg)
        if not template:
            raise ValueError(f"No {kind} pattern configured for split={split}, submode={submode}")

        context = {
            "root": self.cfg_data.cordex.root,
            "folder": self._folder_for_kind(mode_cfg, kind),
            "var": var,
            "gcm": run.gcm,
            "rcm": run.rcm,
            "variant": run.variant,
            "file_suffix": mode_cfg.file_suffix,
            "split": split,
            "submode": submode or "",
        }
        return _format_with_optional_var_suffix(template, context, self.resolver.var_suffix(var, kind))

    def load_lr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        path = self._path(var, run, "lr", split, submode=submode)
        return xr.open_dataset(path)[var]

    def load_hr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        path = self._path(var, run, "hr", split, submode=submode)
        return xr.open_dataset(path)[var]

    def load_time(self, run: RunSpecV2, split: str, submode: Optional[str] = None):
        mode_cfg = self.resolver.mode(split, submode)
        tas_path = self._path("tas", run, "hr", split, submode=submode)
        if os.path.exists(tas_path):
            return xr.open_dataset(tas_path)["time"]

        # Fallback for inputs without tas HR.
        lr_template = self._template("lr", mode_cfg)
        if not lr_template:
            raise FileNotFoundError("Could not resolve time source: no HR tas and no LR template")

        context = {
            "root": self.cfg_data.cordex.root,
            "folder": self._folder_for_kind(mode_cfg, "lr"),
            "var": self.cfg_data.variables_lr[0] if self.cfg_data.variables_lr else self.cfg_data.variables[0],
            "gcm": run.gcm,
            "rcm": run.rcm,
            "variant": run.variant,
            "file_suffix": mode_cfg.file_suffix,
            "split": split,
            "submode": submode or "",
        }
        lr_path = _format_with_optional_var_suffix(
            lr_template,
            context,
            self.resolver.var_suffix(context["var"], "lr"),
        )
        return xr.open_dataset(lr_path)["time"]


class PatternReaderV2(DataReaderV2):
    def __init__(self, cfg: ConfigV2):
        self.cfg = cfg
        self.cfg_data = cfg.data
        self.resolver = _ModeAwarePathResolver(cfg)

    def _template(self, kind: str, mode_cfg: ModeConfigV2) -> str:
        if kind == "lr":
            return mode_cfg.lr_pattern or self.cfg_data.pattern.lr_pattern
        if kind == "hr":
            return mode_cfg.hr_pattern or self.cfg_data.pattern.hr_pattern
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def _folder_for_kind(mode_cfg: ModeConfigV2, kind: str) -> str:
        if kind == "lr":
            return mode_cfg.folder_lr if mode_cfg.folder_lr is not None else mode_cfg.folder
        if kind == "hr":
            return mode_cfg.folder_hr if mode_cfg.folder_hr is not None else mode_cfg.folder
        raise ValueError(f"Unknown kind: {kind}")

    def _resolve_paths(self, var: str, run: RunSpecV2, kind: str, split: str, submode: Optional[str]) -> List[str]:
        mode_cfg = self.resolver.mode(split, submode)
        template = self._template(kind, mode_cfg)
        if not template:
            raise ValueError(f"No {kind} pattern configured for split={split}, submode={submode}")

        context = {
            "root": self.cfg_data.data_dir,
            "folder": self._folder_for_kind(mode_cfg, kind),
            "var": var,
            "gcm": run.gcm or "",
            "rcm": run.rcm or "",
            "variant": run.variant,
            "file_suffix": mode_cfg.file_suffix,
            "split": split,
            "submode": submode or "",
        }

        path_expr = _format_with_optional_var_suffix(template, context, self.resolver.var_suffix(var, kind))
        has_glob = any(tok in path_expr for tok in ["*", "?", "["])
        if has_glob:
            matches = sorted(set(glob.glob(path_expr)))
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {path_expr}")
            if len(matches) > 1 and not self.cfg_data.pattern.allow_multi_file:
                raise ValueError(
                    f"Pattern matched {len(matches)} files for {var} but allow_multi_file=False: {matches}"
                )
            return matches

        if not os.path.exists(path_expr):
            raise FileNotFoundError(path_expr)
        return [path_expr]

    def _open_var(self, paths: Sequence[str], var: str):
        if len(paths) == 1:
            return xr.open_dataset(paths[0])[var]

        arrays = [xr.open_dataset(p)[var] for p in paths]
        return xr.concat(arrays, dim=self.cfg_data.pattern.concat_dim)

    def load_lr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        paths = self._resolve_paths(var, run, "lr", split, submode)
        return self._open_var(paths, var)

    def load_hr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        paths = self._resolve_paths(var, run, "hr", split, submode)
        return self._open_var(paths, var)

    def load_time(self, run: RunSpecV2, split: str, submode: Optional[str] = None):
        probe_var = self.cfg_data.variables[0]
        hr_paths = self._resolve_paths(probe_var, run, "hr", split, submode)
        if hr_paths:
            if len(hr_paths) == 1:
                return xr.open_dataset(hr_paths[0])["time"]
            ds_list = [xr.open_dataset(p) for p in hr_paths]
            return xr.concat([d["time"] for d in ds_list], dim=self.cfg_data.pattern.concat_dim)

        lr_probe = self.cfg_data.variables_lr[0] if self.cfg_data.variables_lr else probe_var
        lr_paths = self._resolve_paths(lr_probe, run, "lr", split, submode)
        if len(lr_paths) == 1:
            return xr.open_dataset(lr_paths[0])["time"]
        ds_list = [xr.open_dataset(p) for p in lr_paths]
        return xr.concat([d["time"] for d in ds_list], dim=self.cfg_data.pattern.concat_dim)


class SinglePairReaderV2(DataReaderV2):
    def __init__(self, cfg: ConfigV2):
        self.cfg = cfg
        self.cfg_data = cfg.data
        self.resolver = _ModeAwarePathResolver(cfg)

    def _file_map(self, kind: str, split: str, submode: Optional[str]) -> Dict[str, str]:
        mode_cfg = self.resolver.mode(split, submode)

        if kind == "lr":
            base_file = self.cfg_data.single_pair.lr_file
            base_map = dict(self.cfg_data.single_pair.lr_files or {})
            override_map = dict(mode_cfg.lr_files or {})
            variables = self.cfg_data.variables_lr or self.cfg_data.variables
        elif kind == "hr":
            base_file = self.cfg_data.single_pair.hr_file
            base_map = dict(self.cfg_data.single_pair.hr_files or {})
            override_map = dict(mode_cfg.hr_files or {})
            variables = self.cfg_data.variables
        else:
            raise ValueError(f"Unknown kind: {kind}")

        out: Dict[str, str] = {}

        if base_file:
            for var in variables:
                out[var] = base_file

        for var, path in base_map.items():
            out[str(var)] = str(path)

        for var, path in override_map.items():
            out[str(var)] = str(path)

        for var, path in list(out.items()):
            out[var] = _append_suffix(path, self.resolver.var_suffix(var, kind))

        return out

    def load_lr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        path_map = self._file_map("lr", split, submode)
        if var not in path_map:
            raise KeyError(f"No LR file configured for var='{var}'")
        return xr.open_dataset(path_map[var])[var]

    def load_hr(self, var: str, run: RunSpecV2, split: str, submode: Optional[str] = None):
        path_map = self._file_map("hr", split, submode)
        if var not in path_map:
            raise KeyError(f"No HR file configured for var='{var}'")
        return xr.open_dataset(path_map[var])[var]

    def load_time(self, run: RunSpecV2, split: str, submode: Optional[str] = None):
        hr_map = self._file_map("hr", split, submode)
        if not hr_map:
            raise ValueError("No HR files configured to infer time")
        first_var = next(iter(hr_map.keys()))
        return xr.open_dataset(hr_map[first_var])["time"]


class PreprocessorV2:
    def __init__(self, cfg_preproc):
        self.cfg = cfg_preproc

    def _maybe_flip_latitude(self, x: torch.Tensor):
        if bool(self.cfg.flip_latitude) and x.ndim >= 3:
            return torch.flip(x, dims=[1])
        return x

    def process_lr(self, x: torch.Tensor, var: str):
        x = self._maybe_flip_latitude(x)
        if not bool(self.cfg.use_pre_normalized_lr):
            x = correct_units(x, var)
        return normalise(
            x,
            var=var,
            mode="lr",
            cfg=self.cfg,
            skip_normalization=bool(self.cfg.use_pre_normalized_lr),
        )

    def process_hr(self, x: torch.Tensor, var: str):
        x = self._maybe_flip_latitude(x)
        if not bool(self.cfg.use_pre_normalized_hr):
            x = correct_units(x, var)
        return normalise(
            x,
            var=var,
            mode="hr",
            cfg=self.cfg,
            skip_normalization=bool(self.cfg.use_pre_normalized_hr),
        )

    def coarsen(self, x: torch.Tensor, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        stride = kernel_size if stride is None else stride
        return torch.nn.functional.avg_pool2d(
            x.unsqueeze(1),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ).squeeze(1)


class DownscalingDatasetV2(Dataset):
    def __init__(
        self,
        reader: DataReaderV2,
        preproc: PreprocessorV2,
        run: RunSpecV2,
        variables_lr: Sequence[str],
        variables_hr: Sequence[str],
        kernel_size_lr: int,
        kernel_size_hr: int,
        split: str,
        submode: Optional[str] = None,
        include_time: bool = True,
        include_year: bool = False,
        return_timepair: bool = False,
        inference_mode: bool = False,
        inference_fill_value: str = "zeros",
    ):
        self.reader = reader
        self.preproc = preproc
        self.run = run
        self.split = split
        self.submode = submode
        self.include_time = include_time
        self.include_year = include_year
        self.return_timepair = return_timepair
        self.inference_mode = inference_mode

        lr_vars: List[torch.Tensor] = []
        for v in variables_lr:
            x = reader.load_lr(v, run, split=split, submode=submode)
            x = torch.from_numpy(x.values).float()
            x = preproc.process_lr(x, v)
            lr_vars.append(x)

        hr_vars: List[torch.Tensor] = []
        hr_coarse_vars: List[torch.Tensor] = []
        if not inference_mode:
            for v in variables_hr:
                y = reader.load_hr(v, run, split=split, submode=submode)
                y = torch.from_numpy(y.values).float()
                if y.ndim != 3:
                    raise ValueError(f"Expected HR tensor (time, y, x), got {tuple(y.shape)} for var='{v}'")
                ny, nx = y.shape[-2], y.shape[-1]
                y = preproc.process_hr(y, v)
                hr_coarse_vars.append(
                    preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_lr, kernel_size_lr, 0).view(y.shape[0], 1, -1)
                )
                if kernel_size_hr > 1:
                    y = preproc.coarsen(y.view(y.shape[0], ny, nx), kernel_size_hr, kernel_size_hr, 0).view(y.shape[0], -1)
                hr_vars.append(y.unsqueeze(1))

        self.x = torch.cat(lr_vars, dim=1)

        if inference_mode:
            n = self.x.shape[0]
            f = max(1, self.x.shape[-1])
            self.z = self._make_target_placeholder(n, f, fill_value=inference_fill_value)
            self.y = self._make_target_placeholder(n, f, fill_value=inference_fill_value)
        else:
            self.z = torch.cat(hr_coarse_vars, dim=1)
            self.y = torch.cat(hr_vars, dim=1)

        if self.include_time or self.include_year:
            time_index = reader.load_time(run, split=split, submode=submode)
            time_feats = self._build_time_features(time_index)
            if time_feats is None:
                raise ValueError("Time features requested but no features were generated")
            if time_feats.shape[0] != self.x.shape[0]:
                raise ValueError(
                    f"Time feature length mismatch: time={time_feats.shape[0]} vs samples={self.x.shape[0]}"
                )
            self.x = torch.cat([self.x, time_feats], dim=1)

        if run.one_hot is not None:
            oh = run.one_hot.expand(self.x.shape[0], -1)
            self.x = torch.cat([self.x, oh], dim=-1)

    def __len__(self):
        if self.return_timepair:
            return self.x.shape[0] - 1
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.return_timepair:
            return (
                self.x[idx],
                self.z[idx],
                self.y[idx],
                self.x[idx + 1],
                self.z[idx + 1],
                self.y[idx + 1],
            )
        return self.x[idx], self.z[idx], self.y[idx]

    @staticmethod
    def _make_target_placeholder(n_samples, n_features, fill_value="zeros"):
        if fill_value == "nan":
            return torch.full((n_samples, 1, n_features), float("nan"), dtype=torch.float32)
        return torch.zeros((n_samples, 1, n_features), dtype=torch.float32)

    def _build_time_features(self, time_index):
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

        feats: List[torch.Tensor] = []
        if self.include_year:
            feats.append(year)
        if self.include_time:
            feats.extend(
                [
                    doy,
                    torch.sin((365 / (2 * math.pi)) * doy),
                    torch.cos((365 / (2 * math.pi)) * doy),
                    torch.sin((365 / math.pi) * doy),
                    torch.cos((365 / math.pi) * doy),
                ]
            )

        if not feats:
            return None
        return torch.cat(feats, dim=1)


def get_ensemble_encoding_scheme_v2(cfg: ConfigV2):
    enc_cfg = cfg.data.ensemble_encoding
    if not isinstance(enc_cfg, dict) or not enc_cfg.get("enabled", False):
        return None

    scheme = str(enc_cfg.get("scheme", "gcm+rcm"))
    valid_schemes = {"gcm", "rcm", "gcm+rcm"}
    if scheme not in valid_schemes:
        raise ValueError(f"Unknown ensemble encoding scheme: {scheme}. Expected one of {sorted(valid_schemes)}")
    return scheme


def build_one_hot_v2(cfg: ConfigV2, gcm_list: Sequence[str], rcm_list: Sequence[str]):
    scheme = get_ensemble_encoding_scheme_v2(cfg)
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


def build_run_specs_v2(cfg: ConfigV2) -> List[RunSpecV2]:
    data_cfg = cfg.data

    if data_cfg.type in {"single_pair", "pattern"}:
        return [RunSpecV2()]

    if data_cfg.type != "cordex_ensemble":
        raise ValueError(f"Unsupported data type: {data_cfg.type}")

    gcm_list, rcm_list, _, _ = get_rcm_gcm_combinations(data_cfg.cordex.root)

    if data_cfg.run_indices is not None:
        indices = [int(i) for i in data_cfg.run_indices]
    elif data_cfg.runs.selection == "first_n":
        indices = list(range(int(data_cfg.runs.n_models)))
    elif data_cfg.runs.selection == "explicit":
        indices = [int(i) for i in data_cfg.runs.indices]
    else:
        raise ValueError(f"Unknown run selection: {data_cfg.runs.selection}")

    one_hot = build_one_hot_v2(cfg, gcm_list, rcm_list)

    runs = []
    for i in indices:
        runs.append(
            RunSpecV2(
                gcm=gcm_list[i],
                rcm=rcm_list[i],
                one_hot=one_hot[i] if one_hot is not None else None,
            )
        )
    return runs


def build_reader_v2(cfg: ConfigV2) -> DataReaderV2:
    dtype = cfg.data.type
    if dtype == "cordex_ensemble":
        return CordexReaderV2(cfg)
    if dtype == "single_pair":
        return SinglePairReaderV2(cfg)
    if dtype == "pattern":
        return PatternReaderV2(cfg)
    raise ValueError(f"Unknown data type: {dtype}")


def build_dataset_v2(
    cfg: ConfigV2,
    split: str,
    submode: Optional[str] = None,
    temporal: Optional[bool] = None,
    inference_mode: bool = False,
):
    reader = build_reader_v2(cfg)
    preproc = PreprocessorV2(cfg.data.preprocessing)
    runs = build_run_specs_v2(cfg)

    if temporal is None:
        temporal = bool(cfg.data.return_timepair)

    return ConcatDataset(
        [
            DownscalingDatasetV2(
                reader=reader,
                preproc=preproc,
                run=run,
                variables_lr=cfg.data.variables_lr or cfg.data.variables,
                variables_hr=cfg.data.variables,
                kernel_size_lr=cfg.data.kernel_size_lr,
                kernel_size_hr=cfg.data.kernel_size_hr,
                split=split,
                submode=submode,
                return_timepair=temporal,
                inference_mode=inference_mode,
            )
            for run in runs
        ]
    )


def get_data_v2(
    cfg: ConfigV2,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    validation_size: Optional[float] = None,
    temporal: Optional[bool] = None,
):
    """
    Build training dataloaders.

    Returns (train_loader, val_loader).
    """
    if batch_size is None:
        batch_size = cfg.training.batch_size

    train_dataset = build_dataset_v2(cfg, split="train", temporal=temporal, inference_mode=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
    )

    val_loader = None
    if validation_size is not None and validation_size > 0:
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))),
            test_size=float(validation_size),
            random_state=cfg.data.random_state,
        )
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
        )
    else:
        try:
            val_dataset = build_dataset_v2(cfg, split="validation", temporal=temporal, inference_mode=False)
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=cfg.training.num_workers,
                )
        except (FileNotFoundError, OSError, KeyError, ValueError):
            val_loader = None

    return train_loader, val_loader


def get_inference_data_v2(
    cfg: ConfigV2,
    split: str,
    submode: Optional[str] = None,
    batch_size: Optional[int] = None,
    temporal: Optional[bool] = None,
    with_targets: bool = False,
):
    """
    Build an inference dataloader for split in {test, inference} with optional user-defined submode.
    """
    if split not in {"test", "inference"}:
        raise ValueError("split must be 'test' or 'inference'")

    if batch_size is None:
        batch_size = cfg.training.batch_size

    dataset = build_dataset_v2(
        cfg,
        split=split,
        submode=submode,
        temporal=temporal,
        inference_mode=not with_targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
