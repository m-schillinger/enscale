# More detailled information on EnScale implementation

## Data folders

Currently, we have separate folders for train and test data (`train`, `test/interpolation`, `test/extrapolation`).
From the `train` data, a random subset (with a fixed random seed, s.t. the split is the same in each call) is taken as the validation set.
"Test loss" in the train files refers to the loss on the validation set.
The actual test data is only used in the inference to generate predictions (and in the separate evaluation as in the paper).

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
- uniform transformation can be combined with logit transformation or "normal" transformation: logit just applies the logit function to map from normalised data (which are probabilities between 0 and 1) to the real line; "normal" transformation applies the inverse of the standard normal CDF to map to the real line, so the resulting data are "normal scores"

## If you want to re-train with your own data

1. Preprocess data or compute normalisation statistics (see above).
2. Adjust dataloading with your paths to the data.

**More will follow.**


## On EnScale training


