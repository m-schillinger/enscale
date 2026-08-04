import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr


DEFAULT_SQRT_VARS = {"pr", "sfcWind"}


def correct_units(data: torch.Tensor, var: str) -> torch.Tensor:
    """Apply the same heuristic unit correction used in the current pipeline."""
    if var == "pr" and data.mean() < 0.1:
        return data * 86400.0
    if var == "tas" and data.mean() > 100.0:
        return data - 273.15
    return data


def parse_subset_specs(specs: Sequence[str]) -> Dict[str, List[str]]:
    """
    Parse repeated --subset args of the form:
        var=glob1,glob2
    Example:
        --subset pr='*CNRM*' --subset tas='*MPI*'
    """
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
    """
    Load subset mapping from JSON file:
    {
      "pr": ["*CNRM*", "*MPI*"],
      "tas": ["*ALADIN*"]
    }
    """
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Subset file must contain a JSON object mapping variable -> list of glob patterns")

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


def classify_domain(file_name: str, hr_regex: re.Pattern, lr_regex: re.Pattern) -> Optional[str]:
    if hr_regex.search(file_name):
        return "hr"
    if lr_regex.search(file_name):
        return "lr"
    return None


def discover_variable_files(
    train_dir: Path,
    variable: str,
    subset_patterns: Optional[Sequence[str]],
) -> List[Path]:
    all_candidates = sorted(train_dir.glob(f"{variable}_day_*.nc"))
    if subset_patterns is None:
        return all_candidates

    matches: List[Path] = []
    for path in all_candidates:
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
            # For fields with extra middle dimensions (e.g., levels), average over those dims.
            reduce_dims = tuple(range(1, arr.ndim - 2))
            arr = arr.mean(dim=reduce_dims)

        arr = torch.flip(arr, dims=[1])
        arr = correct_units(arr, variable)
        tensors.append(arr.float())

    if not tensors:
        raise ValueError(f"No tensors loaded for variable '{variable}'")

    spatial_shapes = {tuple(t.shape[1:]) for t in tensors}
    if len(spatial_shapes) != 1:
        raise ValueError(
            f"Inconsistent spatial shapes for '{variable}': {sorted(spatial_shapes)}. "
            "Please restrict to a compatible subset."
        )

    return torch.cat(tensors, dim=0)


def maybe_sqrt_transform(data: torch.Tensor, variable: str, apply_sqrt: bool, sqrt_vars: set) -> torch.Tensor:
    if apply_sqrt and variable in sqrt_vars:
        if torch.any(data < 0):
            raise ValueError(
                f"Cannot apply sqrt for variable '{variable}' because data contains negative values"
            )
        return torch.sqrt(data)
    return data


def compute_scalar_stats(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mean": torch.mean(data), "std": torch.std(data)}


def compute_pixelwise_stats(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"mean": torch.mean(data, dim=0), "std": torch.std(data, dim=0)}


def compute_uniform_ecdf_matrix(data: torch.Tensor, subsample_size: int, seed: int) -> torch.Tensor:
    """
    Compute per-pixel ECDF quantile matrix for uniform normalization.
    Output shape: [n, H, W], where n == min(subsample_size, len(data)).
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D input [time, H, W], got shape {tuple(data.shape)}")

    n_total = data.shape[0]
    n = min(subsample_size, n_total)
    if n <= 1:
        raise ValueError("subsample_size must resolve to at least 2 samples")

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n_total, generator=rng)[:n]
    sample = data[indices].detach().cpu().numpy()

    # Sort per pixel across time. This is the empirical quantile support.
    sorted_vals = np.sort(sample, axis=0)

    p_source = np.linspace(1.0 / n, 1.0, n, dtype=np.float64)
    p_target = np.linspace(1.0 / (n + 1), n / (n + 1), n, dtype=np.float64)

    flat_sorted = sorted_vals.reshape(n, -1)
    flat_out = np.empty_like(flat_sorted)

    for idx in range(flat_sorted.shape[1]):
        flat_out[:, idx] = np.interp(p_target, p_source, flat_sorted[:, idx])

    out = flat_out.reshape(sorted_vals.shape)
    return torch.from_numpy(out)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stats_file_name(
    domain: str,
    method: str,
    variable: str,
    sqrt_tag: str,
    scope_tag: str = "all",
    subsample_size: Optional[int] = None,
) -> str:
    if method == "uniform":
        return f"{domain}_norm_uniform_ecdf_{variable}_{scope_tag}_subsample-{subsample_size}{sqrt_tag}.pt"
    if method == "normalize_scalar":
        return f"{domain}_norm_scalar_{variable}_{scope_tag}{sqrt_tag}.pt"
    if method == "normalize_pw":
        return f"{domain}_norm_pw_{variable}_{scope_tag}{sqrt_tag}.pt"
    raise ValueError(f"Unknown method for naming: {method}")


def save_artifact(obj, out_dir: Path, file_name: str) -> Path:
    ensure_dir(out_dir)
    path = out_dir / file_name
    torch.save(obj, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generalized normalization-stats computation from a train folder. "
            "Computes scalar/pixelwise stats for HR/LR and HR-only uniform ECDF stats."
        )
    )
    parser.add_argument("--train-dir", type=str, required=True, help="Folder with train NetCDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder to write stats files")

    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variables to process (default: infer from filenames)",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["normalize_scalar", "normalize_pw", "uniform"],
        default=["normalize_scalar", "normalize_pw", "uniform"],
        help="Normalization stats to compute",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["hr", "lr"],
        default=["hr", "lr"],
        help="Domains to compute for (uniform only applies to hr)",
    )

    parser.add_argument(
        "--sqrt-modes",
        nargs="+",
        choices=["no_sqrt", "sqrt"],
        default=["no_sqrt", "sqrt"],
        help="Whether to compute non-sqrt and/or sqrt variants",
    )
    parser.add_argument(
        "--sqrt-vars",
        nargs="+",
        default=sorted(DEFAULT_SQRT_VARS),
        help="Variables eligible for sqrt transform when sqrt mode is enabled",
    )

    parser.add_argument(
        "--subsample-size",
        type=int,
        default=30000,
        help="Subsample size for uniform ECDF computation (HR only)",
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
        "--hr-domain-regex",
        type=str,
        default=r"EUR-11|ALPS_cordexgrid",
        help="Regex used to classify files as HR",
    )
    parser.add_argument(
        "--lr-domain-regex",
        type=str,
        default=r"EUROPE_g025",
        help="Regex used to classify files as LR",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--dry-run", action="store_true", help="Resolve files and planned outputs only")

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.output_dir)

    if not train_dir.exists() or not train_dir.is_dir():
        raise ValueError(f"Invalid --train-dir: {train_dir}")

    hr_regex = re.compile(args.hr_domain_regex)
    lr_regex = re.compile(args.lr_domain_regex)

    subset_map = load_subset_file(args.subset_file)
    subset_map.update(parse_subset_specs(args.subset))

    all_nc_names = sorted([p.name for p in train_dir.glob("*.nc")])
    variables = args.variables if args.variables is not None else infer_variables_from_files(all_nc_names)

    if not variables:
        raise ValueError("No variables found. Provide --variables explicitly or check train-dir contents")

    sqrt_vars = set(args.sqrt_vars)

    manifest = {
        "train_dir": str(train_dir),
        "output_dir": str(out_dir),
        "variables": variables,
        "methods": args.methods,
        "domains": args.domains,
        "sqrt_modes": args.sqrt_modes,
        "sqrt_vars": sorted(sqrt_vars),
        "subsample_size": args.subsample_size,
        "files": {},
        "outputs": [],
    }

    for var in variables:
        var_files = discover_variable_files(
            train_dir=train_dir,
            variable=var,
            subset_patterns=subset_map.get(var),
        )

        if not var_files:
            raise ValueError(f"No files found for variable '{var}'")

        domain_files: Dict[str, List[Path]] = {"hr": [], "lr": []}
        unknown = []
        for fp in var_files:
            domain = classify_domain(fp.name, hr_regex, lr_regex)
            if domain is None:
                unknown.append(fp)
                continue
            domain_files[domain].append(fp)

        if unknown:
            print(
                f"[warn] Skipping {len(unknown)} unclassified files for {var}: "
                f"{[p.name for p in unknown][:5]}"
            )

        manifest["files"][var] = {
            "hr": [str(p) for p in domain_files["hr"]],
            "lr": [str(p) for p in domain_files["lr"]],
        }

        for domain in args.domains:
            files = domain_files[domain]
            if not files:
                print(f"[info] No {domain.upper()} files for variable '{var}', skipping")
                continue

            for sqrt_mode in args.sqrt_modes:
                apply_sqrt = sqrt_mode == "sqrt"
                if apply_sqrt and var not in sqrt_vars:
                    # Avoid duplicate outputs for variables that are never sqrt-transformed.
                    continue
                sqrt_tag = "_sqrt" if apply_sqrt and var in sqrt_vars else ""

                if args.dry_run:
                    if "normalize_scalar" in args.methods:
                        fname = stats_file_name(domain, "normalize_scalar", var, sqrt_tag)
                        manifest["outputs"].append(str(out_dir / fname))

                    if "normalize_pw" in args.methods:
                        fname = stats_file_name(domain, "normalize_pw", var, sqrt_tag)
                        manifest["outputs"].append(str(out_dir / fname))

                    if "uniform" in args.methods and domain == "hr":
                        fname = stats_file_name(
                            domain,
                            "uniform",
                            var,
                            sqrt_tag,
                            subsample_size=args.subsample_size,
                        )
                        manifest["outputs"].append(str(out_dir / fname))
                    continue

                print(f"[info] Loading {len(files)} files for {var} ({domain.upper()})")
                data_raw = load_and_stack_variable(files, var)
                data = maybe_sqrt_transform(data_raw, var, apply_sqrt, sqrt_vars)

                if "normalize_scalar" in args.methods:
                    stats_scalar = compute_scalar_stats(data)
                    fname = stats_file_name(domain, "normalize_scalar", var, sqrt_tag)
                    out_path = save_artifact(stats_scalar, out_dir, fname)
                    manifest["outputs"].append(str(out_path))

                if "normalize_pw" in args.methods:
                    stats_pw = compute_pixelwise_stats(data)
                    fname = stats_file_name(domain, "normalize_pw", var, sqrt_tag)
                    out_path = save_artifact(stats_pw, out_dir, fname)
                    manifest["outputs"].append(str(out_path))

                if "uniform" in args.methods and domain == "hr":
                    ecdf_matrix = compute_uniform_ecdf_matrix(data, args.subsample_size, args.seed)
                    fname = stats_file_name(
                        domain,
                        "uniform",
                        var,
                        sqrt_tag,
                        subsample_size=min(args.subsample_size, data.shape[0]),
                    )
                    out_path = save_artifact(ecdf_matrix, out_dir, fname)
                    manifest["outputs"].append(str(out_path))

    ensure_dir(out_dir)
    manifest_path = out_dir / "norm_stats_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Wrote manifest: {manifest_path}")
    print(f"[done] Planned/generated artifacts: {len(manifest['outputs'])}")


if __name__ == "__main__":
    main()
