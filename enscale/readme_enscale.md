# If you want to use the pre-trained EnScale model

## Data preprocessing
- Check the dataloading options below and convert your predictor files accordingly.
- Convert variables to the following units:
    - temperature: Celsius
    - precipitation:  mm/day
    - rsds: W/m^2
    - sfcWind: m/s
    - psl: Pa
- grid: regular 2.5 degree lonlat grid (see Zenodo) *to do: upload grid to zenodo*
- select region: Process with cdo -sellonlatbox,320,50,25,75 or select the equivalent region otherwise.

Note: EnScale currently needs only all five variables.

Put your predictors in a folder; for more details on dataloading, see below.

## Checkpoints and helpers

- Find checkpoints on Zenodo. There are five checkpoints for the five stages of the EnScale framework.
- In addition also on zenodo: Statistics for standardization. Download these and then adjust the path in the config file.

## Relevant files

- `eval_hierarchical.py`: File to actually create the predictions.
- In the folder `configs_enscale_published`:
    - `config_inference_hierarchical.yaml`: actual config for inference
    - `config_train*`: train configs; these can be left untouched if only inference is done

Adjust `config_inference_hierarchical.yaml` for your needs.
- choose `test` or `inference` mode
    - For `inference.split: test`, ground-truth HR targets are expected in the dataloader setup (for evaluation workflows).
    - For `inference.split: inference`, the loader can run without HR targets; target placeholders are used internally and only model predictions are written.
- put folder name / path to predictors in the 
- optional: you can have several submodes for testing / inference (e.g. different periods / different model groups etc.)


## Interpreting the output

When running hierarchical inference (via `eval_hierarchical_v2.py`), predictions are saved as PyTorch tensors.

Output location:
- root is selected from config via `general.server`:
    - `ada` -> `general.save_dir_root_local`
    - `euler` -> `general.save_dir_root_server`
- final output directory is:
    - `{root}/{inference.output_subdir}/{inference.split}/{inference.submode_or_default}/`
    - where `inference.submode_or_default` is the selected submode, or `default` if none is set.

Saved files:
- `manifest.pt`: metadata dictionary with run info, selected split/submode, sample size, and resolved checkpoint paths per stage.
- `run_idx{K}.pt`: predictions for run index `K` (one file per selected run).

Prediction tensor format (`run_idx{K}.pt`):
- shape: `[N, V, P, S]`
    - `N`: number of timesteps/samples in the selected dataset split
    - `V`: number of target variables (`len(data.variables)`)
    - `P`: number of HR pixels per variable (flattened spatial dimension)
    - `S`: number of stochastic samples (`inference.sample_size`)
- values are saved after unnormalization, in physical units.


## Potential caveats
- The current ECDF transform only generates data within the range of the used training data. 
- We plan to provide also checkpoints for another EnScale model with simpler standardization to overcome this limitation.

# Dataloading options

We've implemented several options for dataloading.
- `cordex_ensemble`: load one run (or several runs) using CORDEX naming patterns with `{gcm}` and `{rcm}` placeholders.
- `single_pair`: load directly from one LR/HR file pair (or one file per variable).
- `pattern`: load by generic glob/path templates, optionally concatenating multiple matched files.

All options use the same core `data` config fields:
- `variables`: HR target variables to predict.
- `variables_lr`: LR input variables. If omitted, `variables` are used.
- `kernel_size_lr`, `kernel_size_hr`: optional coarsening setup.
- `train` and `validation`: mode-specific folder/pattern/file overrides for training/validation.
- `preprocessing`: unit correction + normalization settings.

For inference/test, split-specific dataloading is configured in `inference.data_modes` (`test` and `inference` groups with `default` and optional `submodes`).

## Folder setup by workflow

### A) Run only inference (use a pre-trained model)

Typical setup:
- one folder for the split you evaluate as `test` (optional)
- one folder for custom deployment data as `inference`
- optional subfolders if you want multiple scenarios/regions via `submodes`

Note: in the test mode, it is required to also have ground truth high-res data.
Inference mode is desgined for cases where only predictors are availabe, targets are set to zero.

Example layout:
- `my_data/test/default/`
- `my_data/test/interpolation_2030s/`
- `my_data/test/extrapolation_2090s/`
- `my_data/inference/default/`
- `my_data/inference/new_region_a/`

How to map this in config:
- set `inference.split` to `test` or `inference`
- set `inference.submode` if needed
- define `inference.data_modes.test.default.folder` and optional `inference.data_modes.test.submodes.*.folder`
- define `inference.data_modes.inference.default.folder` and optional `inference.data_modes.inference.submodes.*.folder`

The final filepath is built from your selected reader and pattern settings, for example:
- CORDEX reader: `data.cordex.root` + folder + `data.cordex.*_pattern`
- pattern reader: `data.data_dir` + folder + `data.pattern.*_pattern`

### B) Train your own model

Typical setup:
- `train` folder required
- `valid` folder optional (if not provided, validation can be sampled from train via `validation_size`)
- `test` and `inference` folders optional for later evaluation/deployment

Example layout:
- `my_data/train/`
- `my_data/valid/` (optional)
- `my_data/test/default/` (optional)
- `my_data/inference/default/` (optional)

How to map this in config:
- training uses `data.train` and `data.validation`
- inference/evaluation later uses `inference.data_modes`
- if LR and HR are stored separately, use `folder_lr` and `folder_hr`

## 1) `cordex_ensemble`

Use this when your files follow CORDEX run naming and you want model-run selection/ensemble encoding.

Required fields:
- `data.type: cordex_ensemble`
- `data.cordex.root`
- `data.cordex.lr_pattern`
- `data.cordex.hr_pattern`
- `data.runs` (`selection: first_n|explicit`)

Supported placeholders in pattern strings:
- `{root}`, `{folder}`, `{var}`, `{gcm}`, `{rcm}`, `{variant}`, `{file_suffix}`, `{split}`, `{submode}`, `{var_suffix}`

Mode overrides (`data.train` / `data.validation`) can set:
- `folder`
- `folder_lr`, `folder_hr`
- `file_suffix`
- `lr_pattern`, `hr_pattern`

Template: `config_train_v2_example_cordex_ensemble.yaml`

## 2) `single_pair`

Use this when data is already assembled in fixed files.

Required fields:
- `data.type: single_pair`
- either `data.single_pair.lr_file` + `data.single_pair.hr_file`
- or `data.single_pair.lr_files` + `data.single_pair.hr_files` mappings by variable name

Notes:
- If `lr_file`/`hr_file` are set, they are used as defaults for all variables.
- Per-variable entries in `lr_files`/`hr_files` override the defaults.
- `data.train`/`data.validation` can also override with mode-local `lr_files`/`hr_files`.

Template: `config_train_v2_example_single_pair.yaml`

## 3) `pattern`

Use this when your files do not follow CORDEX naming but can be addressed via string patterns or glob expressions.

Required fields:
- `data.type: pattern`
- `data.data_dir`
- `data.pattern.lr_pattern`
- `data.pattern.hr_pattern`

Pattern behavior:
- Supports placeholders: `{root}`, `{folder}`, `{var}`, `{file_suffix}`, `{split}`, `{submode}`, `{var_suffix}`.
- If pattern contains glob tokens (`*`, `?`, `[]`), all matches are loaded.
- If multiple files are matched, they are concatenated along `data.pattern.concat_dim`.
- Set `data.pattern.allow_multi_file: false` to enforce single-match behavior.

Template: `config_train_v2_example_pattern.yaml`

## Variable-specific filename suffixes

All reader types support suffix injection before file extension:
- `data.lr_var_suffix_default`, `data.hr_var_suffix_default`
- `data.lr_var_suffix_by_var`, `data.hr_var_suffix_by_var`

If `{var_suffix}` is present in a pattern, it is substituted directly. Otherwise, the suffix is inserted before the extension automatically.

# Training your own EnScale model

## Data folders

Recommended split layout:
- required: train
- optional: valid
- optional: test
- optional: inference

For training:
- `data.train.folder` (or `data.train.folder_lr` / `data.train.folder_hr`) should point to training files
- `data.validation.folder` (or `folder_lr` / `folder_hr`) should point to validation files if you keep a separate validation dataset

For inference/evaluation:
- use `inference.data_modes.test.*` for benchmark-style test datasets where also ground-truth HR data is available
- use `inference.data_modes.inference.*` for deployment/custom datasets, for cases where only predictors are availabel

If you do not have a separate validation folder:
- keep `data.validation` unset or not used
- use `validation_size > 0` in training to split from the training dataset randomly

## Data normalisation

We implemented multiple options to pre-process the data before training the model.
See helpers `normalise` and `unnormalise` in `utils.py`.
Note: These require computation of normalisation statistics separately (which are then loaded from disk). 
In same cases, also normalising the data is slow, so we normalise the data once, save it on disk and then load the normalised data directly.

Normalisation options (all of them are done separately for each climate variable):
- primitive: simple scaling of entire field by fixed constants
- normalise_pw: scaling pointwise, i.e., subtract mean and divide by std for each location separately (where mean and std for each location are pre-computed and saved on disk)
- normalise_scalar: scaling of the entire field by mean and std
- uniform: transformation to uniform pointwise; load pre-computed ECDF from disk and interpolate data to it

**Notes on uniform distribution:**
- normalisation is costly, so we did this in advance and then only load the normalised data from disk. `normalise` for `norm_method = "uniform"` is not called during training
- the config files support loading normalized data from data, this skips applying the normalization in dataloading
- we provide a script to pre-normalize data to uniform and save it on disk (see below)
- ECDF computation can be done only on a subset of the data, this makes computations faster, but potentially decreases the accuracy of the transformation
- uniform transformation can be combined with logit transformation or "normal" transformation: logit just applies the logit function to map from normalised data (which are probabilities between 0 and 1) to the real line; "normal" transformation applies the inverse of the standard normal CDF to map to the real line, so the resulting data are "normal scores". This is applied *after* loading the data from disk.

### Computing normalization statistics:

The normalization statistics are generated with two dedicated scripts:
- `compute_norm_stats_general.py`: computes stats from train data folders by variable, with automatic discovery or per-variable subset filtering.
- `compute_uniform_prenorm_data.py`: turns HR data into pre-normalized uniform fields using saved ECDF stats. Note: For flexibility, it does not include the logit/normal transform.

Both scripts keep the existing payload formats used by training:
- `normalize_scalar`: mean/std over the whole field
- `normalize_pw`: mean/std per pixel
- `uniform`: ECDF matrix for HR data only

Supported options in `compute_norm_stats_general.py`:
- HR and LR data
- `normalize_scalar`, `normalize_pw`, and `uniform`
- with and without sqrt
- HR uniform subsampling via `--subsample-size` (i.e. compute )

Supported options in `compute_uniform_prenorm_data.py`:
- HR data only
- 1/(n+1) inverse ECDF transform
- flipped ECDF matrix back to xarray orientation
- precip zero handling via `--zero-mode constant|random_ties|random_image`

**More will follow.**

### EnScale training

In the paper, the EnScale framework consists of several steps. *Currently, you need a separate config file for each step.* 

- The config files for the several super-resolution steps can be identical up to the kernel size parameters (kernel_size_lr, kernel_size_hr).
- But for full flexibility, we still keep the config files separately, in case one wants to adjust hyperparameters for each stage. E.g. often a smaller batch size is needed for later stages due to larger memory constraints, and later models converge after fewer epochs already.

### Notes on config parameters

