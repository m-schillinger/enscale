# EnScale

This repository contains implementations for the paper "EnScale: Temporally-consistent multivariate generative downscaling via proper scoring rules" by M. Schillinger, M. Samarin, X. Shen, R. Knutti, N. Meinshausen. 
**Note:** This is a preliminary status and this repo will be cleaned up and updated further. In case of questions, please contact Maybritt (maybritt.schillinger@stat.math.ethz.ch).

## Folder enscale

Implementation of EnScale.

### Used pre-trained EnScale model

**Workflow**

- download checkpoints (**to do**)
- download normalization statistics
- create a config file for inference
- generate samples from EnScale with `eval_hierarchical_v2.py`

For more details, see `readme_enscale.md` in the folder `enscale`.

### Re-train your own EnScale model

**Workflow**
- preprocess data to match the supported formats
- (optional) compute statistics for standardization of the data
- create config files, see examples 
- train several steps of EnScale separately (`train_only-super_multivariate.py` and `train_only-coarse.py`, optionally `train_only-super_multivariate_temporal-v2.py`)
- create a separate config file for inference
- generate samples from EnScale with `eval_hierarchical_v2.py`
- these can then be used in the evaluation scripts.

For more details, see `readme_enscale.md` in the folder `enscale`.

## Folder benchmarks

Simple baselines from the paper (analogues, EasyUQ).

## Folder evaluation

Functions to calculate evaluation metrics.
A script to run them in an automated fashion is given in `run_evals_script_parallel.py` and one call of this script to reproduce the results of the paper is in `run_eval_command.sh`.
Some metrics which are calculated in each grid point as well (for Fig. 9 & Fig. 11), are not included in the above script and are calculated separately in `run_evals_script_metrics_per_gridpoint.py`.

## Folder plotting

Scripts / notebooks to generate plots in the paper. Uses outputs from the evaluation scripts.

**Note:** Some plots from the most recent version of the paper are still missing.

## Folder conda

Two .yml files for conda environments:
- environment_reproduce_preprint.yml: exported automatically, for exact reproducibility of results in the paper
- environment_modern.yml: reduced package list with more recent python version, recommended for users

Both environments yield identical results (up to numerical noise).

## Folder enscale_review_version

Backup of copy code which was used to create the 
