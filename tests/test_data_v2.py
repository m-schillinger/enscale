import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "enscale"))

from config_v2 import ConfigV2
from data_v2 import PatternReaderV2, RunSpecV2, SinglePairReaderV2


def _write_var_file(path: Path, var: str, start: int, n_time: int = 2):
    t = pd.date_range("2000-01-01", periods=n_time, freq="D")
    vals = np.arange(start, start + n_time * 4).reshape(n_time, 2, 2).astype(np.float32)
    ds = xr.Dataset({var: (("time", "y", "x"), vals)}, coords={"time": t, "y": [0, 1], "x": [0, 1]})
    ds.to_netcdf(path)


def test_pattern_reader_v2_multi_file_concat(tmp_path):
    data_dir = tmp_path / "pattern"
    data_dir.mkdir(parents=True)

    _write_var_file(data_dir / "tas_1950.nc", "tas", start=0)
    _write_var_file(data_dir / "tas_1951.nc", "tas", start=100)

    doc = {
        "data": {
            "type": "pattern",
            "variables": ["tas"],
            "variables_lr": ["tas"],
            "data_dir": str(data_dir),
            "pattern": {
                "lr_pattern": "{root}/{var}_*.nc",
                "hr_pattern": "{root}/{var}_*.nc",
                "allow_multi_file": True,
                "concat_dim": "time",
            },
            "train": {"folder": "", "file_suffix": ""},
            "preprocessing": {
                "norm_method_input": "none",
                "norm_method_output": "none",
            },
        },
    }

    cfg = ConfigV2(**doc)
    reader = PatternReaderV2(cfg)
    arr = reader.load_lr("tas", RunSpecV2(), split="train")

    assert arr.shape[0] == 4
    assert float(arr.values[0, 0, 0]) == 0.0
    assert float(arr.values[2, 0, 0]) == 100.0


def test_single_pair_reader_v2_per_variable_files(tmp_path):
    data_dir = tmp_path / "single_pair"
    data_dir.mkdir(parents=True)

    tas_lr = data_dir / "tas_lr.nc"
    pr_lr = data_dir / "pr_lr.nc"
    tas_hr = data_dir / "tas_hr.nc"
    pr_hr = data_dir / "pr_hr.nc"

    _write_var_file(tas_lr, "tas", start=1)
    _write_var_file(pr_lr, "pr", start=10)
    _write_var_file(tas_hr, "tas", start=20)
    _write_var_file(pr_hr, "pr", start=30)

    doc = {
        "data": {
            "type": "single_pair",
            "variables": ["tas", "pr"],
            "variables_lr": ["tas", "pr"],
            "single_pair": {
                "lr_files": {"tas": str(tas_lr), "pr": str(pr_lr)},
                "hr_files": {"tas": str(tas_hr), "pr": str(pr_hr)},
            },
            "preprocessing": {
                "norm_method_input": "none",
                "norm_method_output": "none",
            },
        },
    }

    cfg = ConfigV2(**doc)
    reader = SinglePairReaderV2(cfg)

    tas_lr_arr = reader.load_lr("tas", RunSpecV2(), split="train")
    pr_lr_arr = reader.load_lr("pr", RunSpecV2(), split="train")
    tas_hr_arr = reader.load_hr("tas", RunSpecV2(), split="train")
    pr_hr_arr = reader.load_hr("pr", RunSpecV2(), split="train")

    assert float(tas_lr_arr.values[0, 0, 0]) == 1.0
    assert float(pr_lr_arr.values[0, 0, 0]) == 10.0
    assert float(tas_hr_arr.values[0, 0, 0]) == 20.0
    assert float(pr_hr_arr.values[0, 0, 0]) == 30.0
