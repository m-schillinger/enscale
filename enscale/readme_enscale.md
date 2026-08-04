# More detailled information on EnScale implementation

## Dataloading options

We've implemented several options for dataloading.
- Single files
- 

**need config templates**

## Data folders

Currently, we have separate folders for train and test data (`train`, `test/interpolation`, `test/extrapolation`).
From the `train` data, a random subset (with a fixed random seed, s.t. the split is the same in each call) is taken as the validation set.
"Test loss" in the train files refers to the loss on the validation set.
The actual test data is only used in the inference to generate predictions (and in the separate evaluation as in the paper).

**add note on ``what if only train''**

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

The normalization statistics are generated with two dedicated scripts:
- `compute_norm_stats_general.py`: computes stats from train data folders by variable, with automatic discovery or per-variable subset filtering.
- `compute_uniform_prenorm_data.py`: turns HR data into pre-normalized uniform fields using saved ECDF stats.

Both scripts keep the existing payload formats used by training:
- `normalize_scalar`: mean/std over the whole field
- `normalize_pw`: mean/std per pixel
- `uniform`: ECDF matrix for HR data only

Supported options in `compute_norm_stats_general.py`:
- HR and LR data
- `normalize_scalar`, `normalize_pw`, and `uniform`
- with and without sqrt
- HR uniform subsampling via `--subsample-size`

Supported options in `compute_uniform_prenorm_data.py`:
- HR data only
- 1/(n+1) inverse ECDF transform
- flipped ECDF matrix back to xarray orientation
- precip zero handling via `--zero-mode constant|random_ties|random_image`

**Notes on uniform distribution:**
- normalisation is costly, so we did this in advance and then only load the normalised data from disk. `normalise` for `norm_method = "uniform"` is not called during training
- uniform transformation can be combined with logit transformation or "normal" transformation: logit just applies the logit function to map from normalised data (which are probabilities between 0 and 1) to the real line; "normal" transformation applies the inverse of the standard normal CDF to map to the real line, so the resulting data are "normal scores"

## If you want to re-train with your own data

1. Preprocess data or compute normalisation statistics with `compute_norm_stats_general.py`.
2. If you use uniform normalization, generate pre-normalised HR fields with `compute_uniform_prenorm_data.py`.
3. Adjust dataloading with your paths to the data.

**More will follow.**

## On EnScale training


