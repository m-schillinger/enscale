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
    save_dir_root_local: str = "results"
    save_dir_root_server: str = ""
    save_dir_pattern: Optional[str] = None

    def __post_init__(self):
        self.n_visual = int(self.n_visual)
        self.seed = int(self.seed)
        self.print_every_nepoch = int(self.print_every_nepoch)
        self.sample_every_nepoch = int(self.sample_every_nepoch)
        self.server = str(self.server)
        self.save_name = str(self.save_name)
        self.resume_epoch = int(self.resume_epoch)
        self.save_dir_root_local = str(self.save_dir_root_local)
        self.save_dir_root_server = str(self.save_dir_root_server)
        if self.save_dir_pattern is not None:
            self.save_dir_pattern = str(self.save_dir_pattern)


# -------------------------
# Data loading
# -------------------------

@dataclass
class DataConfig:
    type: str = "cordex_ensemble"
    variables: List[str] = field(default_factory=lambda: ["pr"])
    variables_lr: Optional[List[str]] = None
    preprocessing: 'DataPreprocessing' = field(default_factory=lambda: DataPreprocessing())
    # runs selection config (maps to previous 'runs' in YAML)
    @dataclass
    class RunsConfig:
        selection: str = "first_n"  # or 'explicit'
        n_models: int = 1
        indices: List[int] = field(default_factory=list)

        def __post_init__(self):
            self.selection = str(self.selection)
            self.n_models = int(self.n_models)
            if self.indices is None:
                self.indices = []
            elif isinstance(self.indices, list):
                self.indices = [int(i) for i in self.indices]

    runs: RunsConfig = field(default_factory=RunsConfig)
    # Cordex-specific path patterns and modes
    @dataclass
    class CordexConfig:
        root: str = ""
        lr_pattern: str = ""
        hr_pattern: str = ""
        modes: dict = field(default_factory=dict)

    cordex: CordexConfig = field(default_factory=CordexConfig)
    # ensemble encoding (kept as mapping to preserve flexibility)
    ensemble_encoding: dict = field(default_factory=dict)
    # single_pair reader mapping (kept generic)
    single_pair: dict = field(default_factory=dict)
    # generic pattern reader config
    patterns: dict = field(default_factory=dict)
    
    kernel_size_lr: int = 1
    stride_lr: Optional[int] = None
    padding_lr: Optional[int] = None
    kernel_size_hr: int = 1
    stride_hr: Optional[int] = None
    padding_hr: Optional[int] = None
    n_models: int = 1
    return_timepair: bool = False
    run_indices: Optional[List[int]] = None
    random_state: int = 42
    validation_source: str = "auto"
    validation_size: Optional[float] = None
    validation_mode: Optional[str] = None
    test_mode: Optional[str] = None
    inference_mode: bool = False
    inference_fill_value: str = "zeros"
    
    precip_zeros: str = "random"
    data_dir: str = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    
    def __post_init__(self):
        # lists
        self.variables = self._coerce_list(self.variables, str)
        self.variables_lr = self._coerce_optional_list(self.variables_lr, str)
        # self.run_indices = self._coerce_list(self.run_indices, int)
        # ints
        # self.n_models = int(self.n_models)
        self.kernel_size_lr = int(self.kernel_size_lr)
        self.kernel_size_hr = int(self.kernel_size_hr)
        self.n_models = int(self.n_models)
        
        self.run_indices = self._coerce_optional_list(self.run_indices, int)
        self.return_timepair = bool(self.return_timepair)
        self.inference_mode = bool(self.inference_mode)
        # floats
        if self.validation_size is not None:
            self.validation_size = float(self.validation_size)
        # strings
        self.validation_source = str(self.validation_source).lower()
        if self.validation_mode is not None:
            self.validation_mode = str(self.validation_mode)
        if self.test_mode is not None:
            self.test_mode = str(self.test_mode)
        self.inference_fill_value = str(self.inference_fill_value).lower()
        self.precip_zeros = str(self.precip_zeros)
        self.data_dir = str(self.data_dir)
        # ensure preprocessing exists
        if isinstance(self.preprocessing, dict):
            self.preprocessing = DataPreprocessing(**self.preprocessing)
        if self.preprocessing is None:
            self.preprocessing = DataPreprocessing()
        # ensure runs exist
        if isinstance(self.runs, dict):
            self.runs = DataConfig.RunsConfig(**self.runs)
        if self.runs is None:
            self.runs = DataConfig.RunsConfig()
        # ensure cordex exists
        if isinstance(self.cordex, dict):
            self.cordex = DataConfig.CordexConfig(**self.cordex)
        if self.cordex is None:
            self.cordex = DataConfig.CordexConfig()
        # ensemble_encoding as dict
        if self.ensemble_encoding is None:
            self.ensemble_encoding = {}
        if not isinstance(self.ensemble_encoding, dict):
            raise ValueError("ensemble_encoding must be a mapping with keys 'enabled' and 'scheme'")
        enabled = bool(self.ensemble_encoding.get("enabled", False))
        scheme = str(self.ensemble_encoding.get("scheme", "gcm+rcm"))
        valid_schemes = {"gcm", "rcm", "gcm+rcm"}
        if enabled and scheme not in valid_schemes:
            raise ValueError(
                f"Unknown ensemble_encoding.scheme: {scheme}. "
                f"Expected one of {sorted(valid_schemes)}"
            )
        self.ensemble_encoding = {"enabled": enabled, "scheme": scheme}
        # single_pair as dict
        if self.single_pair is None:
            self.single_pair = {}
        if self.patterns is None:
            self.patterns = {}

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

    @staticmethod
    def _coerce_optional_int(val):
        if val is None:
            return None
        if isinstance(val, str) and val.lower() == "none":
            return None
        return int(val)

@dataclass
class DataPreprocessing:
    norm_method_input: Optional[str] = None
    norm_method_output: Optional[str] = None
    fft: bool = False
    normal_transform: bool = False
    logit_transform: bool = False
    clip_quantile: Optional[float] = None
    # per-variable sqrt transforms expected as mapping: {var: bool}
    sqrt_transform_out: dict = field(default_factory=dict)
    sqrt_transform_in: dict = field(default_factory=dict)
    sep_mean_std: bool = False
    use_precomputed_hr: bool = False
    normalized_hr_root: str = ""
    normalized_hr_pattern: str = ""
    normalized_hr_modes: dict = field(default_factory=dict)
    normalized_hr_suffix_by_var: dict = field(default_factory=dict)
    normalized_hr_default_suffix: str = ""
    normalized_hr_filter_outliers: bool = False
    # stats and post_transform are nested structures in the YAML
    @dataclass
    class PostTransform:
        logit: bool = False
        gaussian: bool = False

    @dataclass
    class StatsConfig:
        root: str = ""
        pattern: dict = field(default_factory=dict)

    stats: StatsConfig = field(default_factory=StatsConfig)
    post_transform: PostTransform = field(default_factory=PostTransform)

    def __post_init__(self):
        # strings
        if self.norm_method_input is not None:
            self.norm_method_input = str(self.norm_method_input)
        if self.norm_method_output is not None:
            self.norm_method_output = str(self.norm_method_output)
        # bools
        self.fft = bool(self.fft)
        self.sep_mean_std = bool(self.sep_mean_std)
        self.use_precomputed_hr = bool(self.use_precomputed_hr)
        self.normalized_hr_root = str(self.normalized_hr_root)
        self.normalized_hr_pattern = str(self.normalized_hr_pattern)
        self.normalized_hr_filter_outliers = bool(self.normalized_hr_filter_outliers)
        self.normalized_hr_default_suffix = str(self.normalized_hr_default_suffix)
        # dicts: ensure dict types for sqrt transforms and stats.pattern
        if self.sqrt_transform_in is None:
            self.sqrt_transform_in = {}
        if self.sqrt_transform_out is None:
            self.sqrt_transform_out = {}
        if self.normalized_hr_suffix_by_var is None:
            self.normalized_hr_suffix_by_var = {}
        if self.normalized_hr_modes is None:
            self.normalized_hr_modes = {}
        if not isinstance(self.normalized_hr_suffix_by_var, dict):
            raise ValueError("normalized_hr_suffix_by_var must be a dict")
        if not isinstance(self.normalized_hr_modes, dict):
            raise ValueError("normalized_hr_modes must be a dict")
        # coerce nested dicts to dataclasses when loaded from YAML
        if isinstance(self.stats, dict):
            self.stats = DataPreprocessing.StatsConfig(**self.stats)
        if self.stats is None:
            self.stats = DataPreprocessing.StatsConfig()
        if isinstance(self.post_transform, dict):
            self.post_transform = DataPreprocessing.PostTransform(**self.post_transform)
        if self.post_transform is None:
            self.post_transform = DataPreprocessing.PostTransform()
        # Keep old flat keys and nested post_transform in sync.
        self.logit_transform = bool(self.logit_transform or self.post_transform.logit)
        self.post_transform.logit = bool(self.logit_transform)
        self.normal_transform = bool(self.normal_transform or self.post_transform.gaussian)
        self.post_transform.gaussian = bool(self.normal_transform)
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
    p_norm_loss_loc: Optional[List[float]] = field(default_factory=lambda: [4.0])
    p_norm_loss_batch: Optional[List[float]] = field(default_factory=lambda: [4.0])
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
    num_workers: int = 1

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.alpha = float(self.alpha)
        self.burn_in = int(self.burn_in)
        self.save_model_every = int(self.save_model_every)
        self.num_workers = int(self.num_workers)

# -------------------------
# Root config
# -------------------------

@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sparse_layers: SparseLocalLayerConfig = field(default_factory=SparseLocalLayerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    def __post_init__(self):
        # Backwards compatibility: expose top-level `preprocessing` for older code
        self.preprocessing = self.data.preprocessing
        # also expose save_dir roots/pattern at top-level for older callers
        self.save_dir_root_local = self.general.save_dir_root_local
        self.save_dir_root_server = self.general.save_dir_root_server
        self.save_dir_pattern = self.general.save_dir_pattern

