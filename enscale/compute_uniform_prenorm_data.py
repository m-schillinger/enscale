import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import xarray as xr


DEFAULT_SQRT_VARS = {"pr", "sfcWind"}


def correct_units(data: torch.Tensor, var: str) -> torch.Tensor:
    if var == "pr" and data.mean() < 0.1:
        return data * 86400.0
    if var == "tas" and data.mean() > 100.0:
        return data - 273.15
    return data


def parse_subset_specs(specs: Sequence[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --subset spec '{spec}'. Expected format var=pattern1,pattern2")
        var, patterns = spec.split("=", 1)
        var = var.strip()
        pattern_list = [p.strip() for p in patterns.split(",") if p.strip()]
        if not var or not pattern_list:
            raise ValueError(f"Invalid --subset spec '{spec}'. Variable and patterns must be non-empty")
        result[var] = pattern_list
    return result


def load_subset_file(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Subset file must contain a JSON object mapping variable -> list of patterns")
    out: Dict[str, List[str]] = {}
    for var, patterns in raw.items():
        if isinstance(patterns, str):
            out[var] = [patterns]
        elif isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
            out[var] = patterns
        else:
            raise ValueError(f"Invalid subset entry for '{var}'. Expected string or list of strings")
    return out


def infer_variables_from_files(file_names: Sequence[str]) -> List[str]:
    vars_found = set()
    for name in file_names:
        if "_day_" not in name:
            continue
        var = name.split("_day_", 1)[0]
        if var:
            vars_found.add(var)
    return sorted(vars_found)


def discover_variable_files(
    root_dir: Path,
    mode: str,
    variable: str,
    subset_patterns: Optional[Sequence[str]],
) -> List[Path]:
    candidates = sorted((root_dir / mode).glob(f"{variable}_day_*.nc"))
    if subset_patterns is None:
        return candidates
    matches: List[Path] = []
    for path in candidates:
        name = path.name
        if any(path.match(pattern) or re.search(pattern, name) for pattern in subset_patterns):
            matches.append(path)
    return sorted(matches)


def load_and_stack_variable(files: Sequence[Path], variable: str) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for fp in files:
        with xr.open_dataset(fp) as ds:
            if variable not in ds:
                raise KeyError(f"Variable '{variable}' not found in file: {fp}")
            arr = torch.from_numpy(ds[variable].data)

        if arr.ndim < 3:
            raise ValueError(f"Expected at least 3D tensor [time, lat, lon] for {fp}, got shape {tuple(arr.shape)}")

        if arr.ndim > 3:
            reduce_dims = tuple(range(1, arr.ndim - 2))
            arr = arr.mean(dim=reduce_dims)

        arr = torch.flip(arr, dims=[1])
        arr = correct_units(arr, variable)
        tensors.append(arr.float())

    if not tensors:
        raise ValueError(f"No tensors loaded for variable '{variable}'")

    spatial_shapes = {tuple(t.shape[1:]) for t in tensors}
    if len(spatial_shapes) != 1:
        raise ValueError(f"Inconsistent spatial shapes for '{variable}': {sorted(spatial_shapes)}")

    return torch.cat(tensors, dim=0)


def maybe_sqrt_transform(data: torch.Tensor, variable: str, apply_sqrt: bool, sqrt_vars: set) -> torch.Tensor:
    if apply_sqrt and variable in sqrt_vars:
        if torch.any(data < 0):
            raise ValueError(f"Cannot apply sqrt for variable '{variable}' because data contains negative values")
        return torch.sqrt(data)
    return data


def normalize_to_uniform(
    data: torch.Tensor,
    ecdf_matrix: torch.Tensor,
    zero_mode: str,
    seed: int,
) -> torch.Tensor:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data tensor [time, H, W], got shape {tuple(data.shape)}")

    len_full_data = ecdf_matrix.shape[0]
    probs = torch.linspace(1, len_full_data, len_full_data, dtype=torch.float32) / (len_full_data + 1)
    normalized = torch.zeros_like(data, dtype=torch.float32)

    rng = np.random.default_rng(seed)
    time_len = data.shape[0]

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            quantiles = ecdf_matrix[:, i, j]
            data_norm = np.interp(
                data[:, i, j].detach().cpu().numpy(),
                quantiles.detach().cpu().numpy(),
                probs.detach().cpu().numpy(),
            )

            if zero_mode == "constant":
                pass
            elif zero_mode == "random_ties":
                where_zeros = np.where(data_norm == np.min(data_norm))[0]
                data_norm[where_zeros] = data_norm[where_zeros] - rng.uniform(size=len(where_zeros)) * np.min(data_norm)
            elif zero_mode == "random_image":
                unif_sample = rng.uniform(size=time_len)
                where_zeros = np.where(data_norm == np.min(data_norm))[0]
                data_norm[where_zeros] = data_norm[where_zeros] - unif_sample[where_zeros] * np.min(data_norm)
            else:
                raise ValueError(f"Unknown zero handling mode: {zero_mode}")

            normalized[:, i, j] = torch.from_numpy(data_norm.astype(np.float32))

    return normalized


def write_netcdf_like(source_ds: xr.Dataset, data_var: str, out_array: np.ndarray, out_path: Path) -> None:
    out_ds = source_ds.copy(deep=True)
    out_ds[data_var].values = out_array
    try:
        out_ds.to_netcdf(out_path)
    except AttributeError:
        out_ds.to_netcdf(out_path, format="NETCDF3_64BIT")


def load_ecdf_stats(norm_stats_dir: Path, var: str, apply_sqrt: bool) -> torch.Tensor:
    suffix = "_sqrt" if apply_sqrt else ""
    stats_path = norm_stats_dir / f"hr_norm_stats_ecdf_matrix_{var}_train_SUBSAMPLE{suffix}.pt"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing ECDF stats file: {stats_path}")
    ecdf_matrix = torch.load(stats_path, map_location="cpu")
    return torch.flip(ecdf_matrix, dims=[1])


def classify_hr_file(path: Path) -> bool:
    return "EUR-11" in path.name and "ALPS_cordexgrid" in path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pre-normalised HR data for uniform normalization."
    )
    parser.add_argument("--root-dir", type=str, required=True, help="Root data directory containing mode subfolders")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder to write normalized NetCDF files")
    parser.add_argument(
        "--norm-stats-dir",
        type=str,
        required=True,
        help="Folder containing hr_norm_stats_ecdf_matrix_*.pt files",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variables to process (default: infer from filenames)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["train", "test_interpolation", "test_extrapolation"],
        choices=["train", "test_interpolation", "test_extrapolation"],
        help="Modes to process",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        help="Per-variable subset filter: var=pattern1,pattern2 (glob or regex)",
    )
    parser.add_argument(
        "--subset-file",
        type=str,
        default=None,
        help="JSON file mapping variable -> list of patterns",
    )
    parser.add_argument(
        "--sqrt-vars",
        nargs="+",
        default=sorted(DEFAULT_SQRT_VARS),
        help="Variables eligible for sqrt transform before uniform normalization",
    )
    parser.add_argument(
        "--zero-mode",
        choices=["constant", "random_ties", "random_image"],
        default="constant",
        help="How to handle zero-probability ties for precipitation",
    )
    parser.add_argument("--subsample-size", type=int, default=30000, help="ECDF subsample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Resolve files and planned outputs only")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_uniform-pre-normalised.nc",
        help="Suffix appended to output filenames",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    norm_stats_dir = Path(args.norm_stats_dir)
    subset_map = load_subset_file(args.subset_file)
    subset_map.update(parse_subset_specs(args.subset))
    sqrt_vars = set(args.sqrt_vars)

    if not root_dir.exists():
        raise ValueError(f"Invalid --root-dir: {root_dir}")
    if not norm_stats_dir.exists():
        raise ValueError(f"Invalid --norm-stats-dir: {norm_stats_dir}")

    manifest = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "norm_stats_dir": str(norm_stats_dir),
        "variables": args.variables,
        "modes": args.modes,
        "zero_mode": args.zero_mode,
        "subsample_size": args.subsample_size,
        "seed": args.seed,
        "outputs": [],
    }

    # Infer variables from train files if not specified.
    inferred_files = sorted((root_dir / "train").glob("*.nc"))
    variables = args.variables if args.variables is not None else infer_variables_from_files([p.name for p in inferred_files])
    if not variables:
        raise ValueError("No variables found. Provide --variables explicitly or check train directory contents")
    manifest["variables"] = variables

    for var in variables:
        var_subset = subset_map.get(var)
        apply_sqrt = var in sqrt_vars
        ecdf_matrix = load_ecdf_stats(norm_stats_dir, var, apply_sqrt)

        for mode in args.modes:
            files = discover_variable_files(root_dir, mode, var, var_subset)
            files = [fp for fp in files if classify_hr_file(fp)]
            if not files:
                print(f"[info] No HR files found for variable '{var}' in mode '{mode}', skipping")
                continue

            print(f"[info] Processing {len(files)} HR files for '{var}' in mode '{mode}'")
            out_mode_dir = output_dir / mode
            out_mode_dir.mkdir(parents=True, exist_ok=True)

            for fp in files:
                with xr.open_dataset(fp) as source_ds:
                    if var not in source_ds:
                        raise KeyError(f"Variable '{var}' not found in file: {fp}")

                    data = torch.from_numpy(source_ds[var].data).float()
                    if data.ndim != 3:
                        raise ValueError(f"Expected 3D HR data [time, H, W] for {fp}, got shape {tuple(data.shape)}")

                    data = torch.flip(data, dims=[1])
                    data = correct_units(data, var)
                    data = maybe_sqrt_transform(data, var, apply_sqrt, sqrt_vars)
                    out_array = normalize_to_uniform(
                        data=data,
                        ecdf_matrix=ecdf_matrix,
                        zero_mode=args.zero_mode,
                        seed=args.seed,
                    ).detach().cpu().numpy()

                    suffix = "_sqrt" if apply_sqrt else ""
                    out_name = f"{fp.stem}{suffix}{args.output_suffix}"

                    out_path = out_mode_dir / out_name
                    if not args.dry_run:
                        write_netcdf_like(source_ds, var, out_array, out_path)
                    manifest["outputs"].append(str(out_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "uniform_prenorm_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Wrote manifest: {output_dir / 'uniform_prenorm_manifest.json'}")


if __name__ == "__main__":
    main()