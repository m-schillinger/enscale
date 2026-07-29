from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from enscale.archive.config import ModelConfig, LossConfig, SparseLocalLayerConfig, TrainingConfig


@dataclass
class GeneralConfigV2:
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


@dataclass
class ModeConfigV2:
    folder: str = ""
    file_suffix: str = ""
    lr_pattern: Optional[str] = None
    hr_pattern: Optional[str] = None
    lr_files: Dict[str, str] = field(default_factory=dict)
    hr_files: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.folder = str(self.folder)
        self.file_suffix = str(self.file_suffix)
        if self.lr_pattern is not None:
            self.lr_pattern = str(self.lr_pattern)
        if self.hr_pattern is not None:
            self.hr_pattern = str(self.hr_pattern)
        self.lr_files = dict(self.lr_files or {})
        self.hr_files = dict(self.hr_files or {})


@dataclass
class ModeGroupConfigV2:
    default: ModeConfigV2 = field(default_factory=ModeConfigV2)
    submodes: Dict[str, ModeConfigV2] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.default, dict):
            self.default = ModeConfigV2(**self.default)
        if self.default is None:
            self.default = ModeConfigV2()

        coerced: Dict[str, ModeConfigV2] = {}
        for k, v in (self.submodes or {}).items():
            if isinstance(v, ModeConfigV2):
                coerced[str(k)] = v
            else:
                coerced[str(k)] = ModeConfigV2(**(v or {}))
        self.submodes = coerced


@dataclass
class DataPreprocessingV2:
    norm_method_input: Optional[str] = None
    norm_method_output: Optional[str] = None
    fft: bool = False
    normal_transform: bool = False
    logit_transform: bool = False
    clip_quantile: Optional[float] = None
    sqrt_transform_out: Dict[str, bool] = field(default_factory=dict)
    sqrt_transform_in: Dict[str, bool] = field(default_factory=dict)
    sep_mean_std: bool = False

    # Shared pre-normalized behavior for all reader types.
    use_pre_normalized_lr: bool = False
    use_pre_normalized_hr: bool = False

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
        if self.norm_method_input is not None:
            self.norm_method_input = str(self.norm_method_input)
        if self.norm_method_output is not None:
            self.norm_method_output = str(self.norm_method_output)

        self.fft = bool(self.fft)
        self.normal_transform = bool(self.normal_transform)
        self.logit_transform = bool(self.logit_transform)
        self.sep_mean_std = bool(self.sep_mean_std)
        self.use_pre_normalized_lr = bool(self.use_pre_normalized_lr)
        self.use_pre_normalized_hr = bool(self.use_pre_normalized_hr)

        if self.clip_quantile is not None:
            self.clip_quantile = float(self.clip_quantile)

        self.sqrt_transform_in = dict(self.sqrt_transform_in or {})
        self.sqrt_transform_out = dict(self.sqrt_transform_out or {})

        if isinstance(self.stats, dict):
            self.stats = DataPreprocessingV2.StatsConfig(**self.stats)
        if self.stats is None:
            self.stats = DataPreprocessingV2.StatsConfig()

        if isinstance(self.post_transform, dict):
            self.post_transform = DataPreprocessingV2.PostTransform(**self.post_transform)
        if self.post_transform is None:
            self.post_transform = DataPreprocessingV2.PostTransform()

        # Keep flat and nested representations synchronized.
        self.logit_transform = bool(self.logit_transform or self.post_transform.logit)
        self.post_transform.logit = bool(self.logit_transform)
        self.normal_transform = bool(self.normal_transform or self.post_transform.gaussian)
        self.post_transform.gaussian = bool(self.normal_transform)


@dataclass
class DataConfigV2:
    type: str = "cordex_ensemble"  # cordex_ensemble | single_pair | pattern
    variables: List[str] = field(default_factory=lambda: ["pr"])
    variables_lr: Optional[List[str]] = None

    @dataclass
    class RunsConfig:
        selection: str = "first_n"  # first_n | explicit
        n_models: int = 1
        indices: List[int] = field(default_factory=list)

        def __post_init__(self):
            self.selection = str(self.selection)
            self.n_models = int(self.n_models)
            self.indices = [int(i) for i in (self.indices or [])]

    runs: RunsConfig = field(default_factory=RunsConfig)

    @dataclass
    class CordexConfig:
        root: str = ""
        lr_pattern: str = ""
        hr_pattern: str = ""

    cordex: CordexConfig = field(default_factory=CordexConfig)

    # Single-pair can be a single file or per-variable map.
    @dataclass
    class SinglePairConfig:
        lr_file: Optional[str] = None
        hr_file: Optional[str] = None
        lr_files: Dict[str, str] = field(default_factory=dict)
        hr_files: Dict[str, str] = field(default_factory=dict)

        def __post_init__(self):
            if self.lr_file is not None:
                self.lr_file = str(self.lr_file)
            if self.hr_file is not None:
                self.hr_file = str(self.hr_file)
            self.lr_files = dict(self.lr_files or {})
            self.hr_files = dict(self.hr_files or {})

    single_pair: SinglePairConfig = field(default_factory=SinglePairConfig)

    # Generic pattern-based config.
    @dataclass
    class PatternConfig:
        lr_pattern: str = ""
        hr_pattern: str = ""
        allow_multi_file: bool = True
        concat_dim: str = "time"

        def __post_init__(self):
            self.lr_pattern = str(self.lr_pattern)
            self.hr_pattern = str(self.hr_pattern)
            self.allow_multi_file = bool(self.allow_multi_file)
            self.concat_dim = str(self.concat_dim)

    pattern: PatternConfig = field(default_factory=PatternConfig)

    # Training-time mode hierarchy only.
    train: ModeConfigV2 = field(default_factory=ModeConfigV2)
    validation: ModeConfigV2 = field(default_factory=ModeConfigV2)

    ensemble_encoding: dict = field(default_factory=dict)

    kernel_size_lr: int = 1
    kernel_size_hr: int = 1
    n_models: int = 1
    return_timepair: bool = False
    run_indices: Optional[List[int]] = None
    random_state: int = 42

    # Variable-specific suffix injection for pattern outputs.
    lr_var_suffix_by_var: Dict[str, str] = field(default_factory=dict)
    hr_var_suffix_by_var: Dict[str, str] = field(default_factory=dict)
    lr_var_suffix_default: str = ""
    hr_var_suffix_default: str = ""

    # Used mainly by downstream scripts.
    precip_zeros: str = "random"
    data_dir: str = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"

    preprocessing: DataPreprocessingV2 = field(default_factory=DataPreprocessingV2)

    def __post_init__(self):
        self.type = str(self.type)
        self.variables = self._coerce_list(self.variables, str)
        self.variables_lr = self._coerce_optional_list(self.variables_lr, str)

        self.kernel_size_lr = int(self.kernel_size_lr)
        self.kernel_size_hr = int(self.kernel_size_hr)
        self.n_models = int(self.n_models)
        self.return_timepair = bool(self.return_timepair)
        self.random_state = int(self.random_state)

        self.run_indices = self._coerce_optional_list(self.run_indices, int)
        self.precip_zeros = str(self.precip_zeros)
        self.data_dir = str(self.data_dir)

        self.lr_var_suffix_by_var = dict(self.lr_var_suffix_by_var or {})
        self.hr_var_suffix_by_var = dict(self.hr_var_suffix_by_var or {})
        self.lr_var_suffix_default = str(self.lr_var_suffix_default)
        self.hr_var_suffix_default = str(self.hr_var_suffix_default)

        if isinstance(self.preprocessing, dict):
            self.preprocessing = DataPreprocessingV2(**self.preprocessing)
        if self.preprocessing is None:
            self.preprocessing = DataPreprocessingV2()

        if isinstance(self.runs, dict):
            self.runs = DataConfigV2.RunsConfig(**self.runs)
        if self.runs is None:
            self.runs = DataConfigV2.RunsConfig()

        if isinstance(self.cordex, dict):
            self.cordex = DataConfigV2.CordexConfig(**self.cordex)
        if self.cordex is None:
            self.cordex = DataConfigV2.CordexConfig()

        # Support legacy single_pair YAML structure with inputs/targets maps.
        if isinstance(self.single_pair, dict):
            if "inputs" in self.single_pair or "targets" in self.single_pair:
                in_cfg = self.single_pair.get("inputs", {}) or {}
                out_cfg = self.single_pair.get("targets", {}) or {}
                remapped = {
                    "lr_file": in_cfg.get("file"),
                    "hr_file": out_cfg.get("file"),
                    "lr_files": in_cfg.get("files", {}),
                    "hr_files": out_cfg.get("files", {}),
                }
                self.single_pair = DataConfigV2.SinglePairConfig(**remapped)
            else:
                self.single_pair = DataConfigV2.SinglePairConfig(**self.single_pair)
        if self.single_pair is None:
            self.single_pair = DataConfigV2.SinglePairConfig()

        if isinstance(self.pattern, dict):
            self.pattern = DataConfigV2.PatternConfig(**self.pattern)
        if self.pattern is None:
            self.pattern = DataConfigV2.PatternConfig()

        self.train = self._coerce_mode(self.train)
        self.validation = self._coerce_mode(self.validation)
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

    @staticmethod
    def _coerce_list(val: Any, element_type):
        if isinstance(val, list):
            return [element_type(v) for v in val]
        return [element_type(val)]

    @staticmethod
    def _coerce_optional_list(val: Any, element_type):
        if val is None:
            return None
        if isinstance(val, str) and val.lower() == "none":
            return None
        return DataConfigV2._coerce_list(val, element_type)

    @staticmethod
    def _coerce_mode(mode: Any) -> ModeConfigV2:
        if mode is None:
            return ModeConfigV2()
        if isinstance(mode, ModeConfigV2):
            return mode
        if isinstance(mode, dict):
            return ModeConfigV2(**mode)
        raise TypeError(f"Unsupported mode config type: {type(mode)}")

    @staticmethod
    def _coerce_mode_group(group: Any) -> ModeGroupConfigV2:
        if group is None:
            return ModeGroupConfigV2()
        if isinstance(group, ModeGroupConfigV2):
            return group
        if isinstance(group, dict):
            return ModeGroupConfigV2(**group)
        raise TypeError(f"Unsupported mode group config type: {type(group)}")

    def resolve_mode(self, split: str, submode: Optional[str] = None) -> ModeConfigV2:
        split = str(split)

        if split == "train":
            return self.train
        if split in {"valid", "validation", "val"}:
            return self.validation

        raise ValueError(f"Unsupported training split '{split}'")

    def var_suffix(self, var: str, kind: str) -> str:
        if kind == "lr":
            return str(self.lr_var_suffix_by_var.get(var, self.lr_var_suffix_default))
        if kind == "hr":
            return str(self.hr_var_suffix_by_var.get(var, self.hr_var_suffix_default))
        raise ValueError(f"Unknown kind '{kind}'")


@dataclass
class InferenceConfigV2:
    @dataclass
    class DataModes:
        test: ModeGroupConfigV2 = field(default_factory=ModeGroupConfigV2)
        inference: ModeGroupConfigV2 = field(default_factory=ModeGroupConfigV2)

        def __post_init__(self):
            if isinstance(self.test, dict):
                self.test = ModeGroupConfigV2(**self.test)
            if self.test is None:
                self.test = ModeGroupConfigV2()

            if isinstance(self.inference, dict):
                self.inference = ModeGroupConfigV2(**self.inference)
            if self.inference is None:
                self.inference = ModeGroupConfigV2()

        def resolve_mode(self, split: str, submode: Optional[str] = None) -> ModeConfigV2:
            split = str(split)
            if split == "test":
                group = self.test
            elif split == "inference":
                group = self.inference
            else:
                raise ValueError(f"Unsupported inference split '{split}'")

            if submode is None:
                return group.default
            if submode not in group.submodes:
                raise KeyError(
                    f"Unknown {split} submode '{submode}'. Available: {sorted(group.submodes.keys())}"
                )
            return group.submodes[submode]

    # Either "pretrained" or "train_output". None means no inference section was configured.
    checkpoint_source: Optional[str] = None

    # Used when checkpoint_source == pretrained
    pretrained_checkpoints: Dict[str, str] = field(default_factory=dict)

    # Used when checkpoint_source == train_output
    train_run_dir: Optional[str] = None

    # Data selection
    split: str = "test"  # test | inference
    submode: Optional[str] = None
    batch_size: Optional[int] = None
    data_modes: DataModes = field(default_factory=DataModes)

    def __post_init__(self):
        if self.checkpoint_source is not None:
            self.checkpoint_source = str(self.checkpoint_source)
            if self.checkpoint_source not in {"pretrained", "train_output"}:
                raise ValueError("checkpoint_source must be one of: pretrained, train_output")

        self.pretrained_checkpoints = dict(self.pretrained_checkpoints or {})

        if self.train_run_dir is not None:
            self.train_run_dir = str(self.train_run_dir)

        self.split = str(self.split)
        if self.split not in {"test", "inference"}:
            raise ValueError("inference split must be one of: test, inference")

        if self.submode is not None:
            self.submode = str(self.submode)

        if self.batch_size is not None:
            self.batch_size = int(self.batch_size)

        if isinstance(self.data_modes, dict):
            self.data_modes = InferenceConfigV2.DataModes(**self.data_modes)
        if self.data_modes is None:
            self.data_modes = InferenceConfigV2.DataModes()

        if self.checkpoint_source == "pretrained" and not self.pretrained_checkpoints:
            raise ValueError("pretrained_checkpoints must be provided when checkpoint_source='pretrained'")

        if self.checkpoint_source == "train_output" and not self.train_run_dir:
            raise ValueError("train_run_dir must be provided when checkpoint_source='train_output'")


@dataclass
class ConfigV2:
    general: GeneralConfigV2 = field(default_factory=GeneralConfigV2)
    data: DataConfigV2 = field(default_factory=DataConfigV2)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sparse_layers: SparseLocalLayerConfig = field(default_factory=SparseLocalLayerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfigV2 = field(default_factory=InferenceConfigV2)

    def __post_init__(self):
        if isinstance(self.general, dict):
            self.general = GeneralConfigV2(**self.general)
        if self.general is None:
            self.general = GeneralConfigV2()

        if isinstance(self.data, dict):
            self.data = DataConfigV2(**self.data)
        if self.data is None:
            self.data = DataConfigV2()

        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if self.model is None:
            self.model = ModelConfig()

        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if self.loss is None:
            self.loss = LossConfig()

        if isinstance(self.sparse_layers, dict):
            self.sparse_layers = SparseLocalLayerConfig(**self.sparse_layers)
        if self.sparse_layers is None:
            self.sparse_layers = SparseLocalLayerConfig()

        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if self.training is None:
            self.training = TrainingConfig()

        if isinstance(self.inference, dict):
            self.inference = InferenceConfigV2(**self.inference)
        if self.inference is None:
            self.inference = InferenceConfigV2()

        # Backward convenience aliases used in parts of the codebase.
        self.preprocessing = self.data.preprocessing
        self.save_dir_root_local = self.general.save_dir_root_local
        self.save_dir_root_server = self.general.save_dir_root_server
        self.save_dir_pattern = self.general.save_dir_pattern
