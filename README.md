# EnScale

This repository contains implementations for the paper "EnScale: Temporally-consistent multivariate generative downscaling via proper scoring rules" by M. Schillinger, M. Samarin, X. Shen, R. Knutti, N. Meinshausen. 
**Note:** This is a preliminary status and this repo will be cleaned up and updated further. In case of questions, please contact Maybritt (maybritt.schillinger@stat.math.ethz.ch).

## Folder enscale

Implementation of EnScale.

**Workflow**
- train each step of EnScale separately (`train_only-super_multivariate.py` and `train_only-coarse.py`), example calls in `run_super.sh` and `run_coarse.sh`.
- generate samples from EnScale with `eval_multi_step_coarse_from_super.py`, example call in `run_eval.sh`.
- these can then be used in the evaluation scripts.

## Folder benchmarks

Simple baselines from the paper (analogues, EasyUQ).

## Folder evaluation

Functions to calculate evaluation metrics.
A script to run them in an automated fashion is given in `run_evals_script_parallel.py` and one call of this script to reproduce the results of the paper is in `run_eval_command.sh`.
Some metrics which are calculated in each grid point as well (for Fig. 9 & Fig. 11), are not included in the above script and are calculated separately in `run_evals_script_metrics_per_gridpoint.py`.

## Folder plotting

Scripts / notebooks to generate plots in the paper. Uses outputs from the evaluation scripts.

