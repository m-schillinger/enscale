from dataclasses import dataclass

import scipy
from enscale.utils import correct_units
import torch
from typing import Optional

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


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def process_hr(self, x, var):
        if self.cfg.correct_units:
            x = correct_units(x, var)
        if self.cfg.apply_normalisation:
            x = normalise(
                x,
                mode="hr",
                data_type=var,
                sqrt_transform=self.cfg.sqrt_transform_out,
                norm_method=self.cfg.norm_output,
            )
        if self.cf.logit:
            x = torch.logit(x)
            hr_np = x.detach().cpu().numpy()
            hr_np_gauss = scipy.stats.norm.ppf(hr_np) # more stable than torch.Normal.icdf
            x = torch.from_numpy(hr_np_gauss).to(x.dtype).to(x.device)

        return x
    
    def process_lr(self, x, var):
        if self.cfg.correct_units:
            x = correct_units(x, var)
        if self.cfg.apply_normalisation:
            x = normalise(
                x,
                mode="lr",
                data_type=var,
                sqrt_transform=self.cfg.sqrt_transform_in,
                norm_method=self.cfg.norm_input,
            )
        return x

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
            y = preproc.process_hr(y, v)
            hr_vars.append(y)
            hr_coarse_vars.append(
                preproc.coarsen(y, kernel_size, kernel_size, 0)
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

            time_feats = self._expand_time_features(time_feats_1d, self.x)

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

        months_np = time_index.month.values.astype(int)
        days_np   = time_index.day.values.astype(int)
        years_np  = time_index.year.values.astype(float)

        is_leap = is_leap_year(years_np)
        leap_year_mask = is_leap & (months_np == 2) & (days_np == 29)
        consider_leap = bool(np.any(leap_year_mask))

        doy = day_of_year_vectorized(
            months_np,
            days_np,
            is_leap,
            consider_leap=consider_leap,
        )

        doy = torch.from_numpy(doy).float().unsqueeze(1)
        year = torch.from_numpy(years_np).float().unsqueeze(1)

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
