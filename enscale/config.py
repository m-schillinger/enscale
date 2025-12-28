# config.py
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# config.py
from dataclasses import dataclass, field
from typing import List, Optional


# -------------------------
# General / bookkeeping
# -------------------------

@dataclass
class GeneralConfig:
    n_visual: int = 8
    seed: int = 222
    print_every_nepoch: int = 1
    sample_every_nepoch: int = 50
    server: str = "ada"
    save_name: str = ""
    resume_epoch: int = 0

    def __post_init__(self):
        self.n_visual = int(self.n_visual)
        self.seed = int(self.seed)
        self.print_every_nepoch = int(self.print_every_nepoch)
        self.sample_every_nepoch = int(self.sample_every_nepoch)
        self.server = str(self.server)
        self.save_name = str(self.save_name)
        self.resume_epoch = int(self.resume_epoch)


# -------------------------
# Data loading
# -------------------------

@dataclass
class DataConfig:
    variables: List[str] = field(default_factory=lambda: ["pr"])
    variables_lr: Optional[List[str]] = None
    n_models: int = 1
    run_indices: List[int] = field(default_factory=lambda: list(range(8)))
    kernel_size_lr: int = 1
    stride_lr: Optional[int] = None
    padding_lr: Optional[int] = None
    kernel_size_hr: int = 1
    mask_gcm: bool = False
    tr_te_split: str = "random"
    tr_te_split_ratio: float = 0.9
    test_model_index: Optional[int] = None
    train_model_index: Optional[int] = None
    train_run_indices: Optional[List[int]] = None
    test_run_indices: Optional[List[int]] = None
    clip_quantile_data: Optional[float] = None
    only_winter: bool = False
    filter_outliers: bool = False
    precip_zeros: str = "random"
    data_dir: str = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    ignore_one_hot_gcm: bool = False
    ignore_one_hot_rcm: bool = False

    def __post_init__(self):
        # lists
        self.variables = self._coerce_list(self.variables, str)
        self.variables_lr = self._coerce_optional_list(self.variables_lr, str)
        self.run_indices = self._coerce_list(self.run_indices, int)
        # ints
        self.n_models = int(self.n_models)
        self.kernel_size_lr = int(self.kernel_size_lr)
        self.kernel_size_hr = int(self.kernel_size_hr)
        if self.stride_lr is not None:
            self.stride_lr = int(self.stride_lr)
        if self.padding_lr is not None:
            self.padding_lr = int(self.padding_lr)
        if self.test_model_index is not None:
            self.test_model_index = int(self.test_model_index)
        if self.train_model_index is not None:
            self.train_model_index = int(self.train_model_index)
        if self.test_run_indices is not None:
            self.test_run_indices = self._coerce_list(self.test_run_indices, int)
        if self.train_run_indices is not None:
            self.train_run_indices = self._coerce_list(self.train_run_indices, int)
        # floats
        self.tr_te_split_ratio = float(self.tr_te_split_ratio)
        if self.clip_quantile_data is not None:
            self.clip_quantile_data = float(self.clip_quantile_data)
        # strings
        self.tr_te_split = str(self.tr_te_split)
        self.precip_zeros = str(self.precip_zeros)
        self.data_dir = str(self.data_dir)

    @staticmethod
    def _coerce_list(val, element_type):
        if isinstance(val, list):
            return [element_type(v) for v in val]
        return [element_type(val)]

    @staticmethod
    def _coerce_optional_list(val, element_type):
        if val is None:
            return None
        if isinstance(val, str) and val.lower() == "none":
            return None
        return DataConfig._coerce_list(val, element_type)

@dataclass
class DataPreprocessing:
    norm_method_input: Optional[str] = None
    norm_method_output: Optional[str] = None
    fft: bool = False
    logit_transform: bool = False
    normal_transform: bool = False
    clip_quantile: Optional[float] = None
    sqrt_transform_out: bool = False
    sqrt_transform_in: bool = False
    sep_mean_std: bool = False

    def __post_init__(self):
        # strings
        if self.norm_method_input is not None:
            self.norm_method_input = str(self.norm_method_input)
        if self.norm_method_output is not None:
            self.norm_method_output = str(self.norm_method_output)
        # bools
        self.fft = bool(self.fft)
        self.logit_transform = bool(self.logit_transform)
        self.normal_transform = bool(self.normal_transform)
        self.sqrt_transform_out = bool(self.sqrt_transform_out)
        self.sqrt_transform_in = bool(self.sqrt_transform_in)
        self.sep_mean_std = bool(self.sep_mean_std)
        # floats
        if self.clip_quantile is not None:
            self.clip_quantile = float(self.clip_quantile)


# -------------------------
# Model architecture
# -------------------------

@dataclass
class ModelConfig:
    method: str = "eng_unet"
    conv: bool = False
    nicolai_layers: bool = False
    conv_concat: bool = False
    num_noise_channels: int = 1
    conv_dim: int = 16
    hidden_dim: int = 1000
    layer_shrinkage: int = 16
    noise_dim: int = 100
    dropout: bool = False
    noise_std: float = 1.0
    out_act: Optional[str] = None
    num_layer: int = 6
    preproc_layer: bool = False
    preproc_dim: int = 20
    mlp: bool = True
    bn: bool = False
    one_hot_in_super: bool = False
    one_hot_only_in_ups: bool = False
    add_x_in_super: bool = False
    split_coarse_model: bool = False

    def __post_init__(self):
        # strings
        self.method = str(self.method)
        if self.out_act is not None:
            self.out_act = str(self.out_act)
        # bools
        self.conv = bool(self.conv)
        self.nicolai_layers = bool(self.nicolai_layers)
        self.conv_concat = bool(self.conv_concat)
        self.dropout = bool(self.dropout)
        self.preproc_layer = bool(self.preproc_layer)
        self.mlp = bool(self.mlp)
        self.bn = bool(self.bn)
        self.one_hot_in_super = bool(self.one_hot_in_super)
        self.one_hot_only_in_ups = bool(self.one_hot_only_in_ups)
        self.add_x_in_super = bool(self.add_x_in_super)
        self.split_coarse_model = bool(self.split_coarse_model)
        # ints
        self.num_noise_channels = int(self.num_noise_channels)
        self.conv_dim = int(self.conv_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.layer_shrinkage = int(self.layer_shrinkage)
        self.noise_dim = int(self.noise_dim)
        self.num_layer = int(self.num_layer)
        self.preproc_dim = int(self.preproc_dim)
        # floats
        self.noise_std = float(self.noise_std)


# -------------------------
# Losses / constraints
# -------------------------

@dataclass
class LossConfig:
    avg_constraint: bool = False
    max_loss: bool = False
    norm_loss: bool = False
    norm_loss_loc: bool = False
    lambda_norm_loss_loc: float = 1.0
    norm_loss_batch: bool = False
    agg_norm_loss: str = "mean"
    norm_loss_per_var: bool = False
    p_norm_loss_loc: Optional[List[float]] = None
    p_norm_loss_batch: Optional[List[float]] = None
    lambda_coarse: float = 0.5
    beta: float = 1.0
    beta_norm_loss: float = 1.0
    patched_loss: bool = False
    patch_size: int = 8
    calc_raw_loss: bool = False

def __post_init__(self):
        # floats
        self.lambda_norm_loss_loc = float(self.lambda_norm_loss_loc)
        self.lambda_coarse = float(self.lambda_coarse)
        self.beta = float(self.beta)
        self.beta_norm_loss = float(self.beta_norm_loss)
        self.patch_size = int(self.patch_size)

        # optional lists
        self.p_norm_loss_loc = self._coerce_optional_float_list(self.p_norm_loss_loc)
        self.p_norm_loss_batch = self._coerce_optional_float_list(self.p_norm_loss_batch)

@staticmethod
def _coerce_optional_float_list(val):
    if val is None:
        return None
    if isinstance(val, str) and val.lower() == "none":
        return None
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, list):
        return [float(v) for v in val]
    raise TypeError(f"Invalid type for optional float list: {type(val)}")

# -------------------------
# Location-specific layers
# -------------------------

@dataclass
class SparseLocalLayerConfig:
    num_neighbors_res: int = 25
    num_neighbors_ups: int = 9
    latent_dim: int = 12
    mlp_depth: int = 3
    noise_dim_mlp: int = 0
    double_linear: bool = False
    add_intermediate_loss: bool = False
    add_mse_loss: bool = False
    lambda_mse_loss: float = 1.0
    not_split_residuals: bool = False

    def __post_init__(self):
        # ints
        self.num_neighbors_res = int(self.num_neighbors_res)
        self.num_neighbors_ups = int(self.num_neighbors_ups)
        self.latent_dim = int(self.latent_dim)
        self.mlp_depth = int(self.mlp_depth)
        self.noise_dim_mlp = int(self.noise_dim_mlp)
        # bools
        self.double_linear = bool(self.double_linear)
        self.add_intermediate_loss = bool(self.add_intermediate_loss)
        self.add_mse_loss = bool(self.add_mse_loss)
        self.not_split_residuals = bool(self.not_split_residuals)
        # floats
        self.lambda_mse_loss = float(self.lambda_mse_loss)


# -------------------------
# Training
# -------------------------

@dataclass
class TrainingConfig:
    batch_size: int = 512
    num_epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 0.0
    alpha: float = 1.0
    burn_in: int = 0
    save_model_every: int = 50

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.alpha = float(self.alpha)
        self.burn_in = int(self.burn_in)
        self.save_model_every = int(self.save_model_every)

# -------------------------
# Root config
# -------------------------

@dataclass
class Config:
    general: GeneralConfig = GeneralConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    sparse_layers: SparseLocalLayerConfig = SparseLocalLayerConfig()
    training: TrainingConfig = TrainingConfig()
    preprocessing: DataPreprocessing = DataPreprocessing()