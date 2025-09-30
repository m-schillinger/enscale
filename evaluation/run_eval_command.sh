#### interpolation (default)
# one run for just the benchmarks
nice -n 7 python run_evals_script_parallel.py --save_name "full" --subfolder "benchmarks" --run_id "maybritt_3step_dense-conv_v3" -1 -2 -3 -7 -8 -11 -12 -13 -15 -17 --project_specifier maybritt --variables tas pr sfcWind rsds --num_samples 9
# another run for the maybritt methods
nice -n 7 python run_evals_script_parallel.py --save_name "full2" --subfolder "maybritt" --run_id "maybritt_nicolai_zeros-constant_dec-1e-3_onehot" "maybritt_nicolai_zeros-constant_dec-1e-3_onehot_temporal" -1 -2 -3 -7 -8 -11 -12 -13 -15 -17 --project_specifier maybritt --variables tas pr sfcWind rsds --num_samples 9 --not_load_gan --not_load_benchmarks --not_load_diffusion

#### extrapolation, --period 2090-2099
nice -n 7 python run_evals_script_parallel.py --save_name "full" --subfolder "benchmarks" --run_id "maybritt_3step_dense-conv_v3" -1 -2 -3 -7 -8 -11 -12 -13 -15 -17 --project_specifier maybritt --variables tas pr sfcWind rsds --num_samples 9 --period 2090-2099
nice -n 7 python run_evals_script_parallel.py --save_name "full2" --subfolder "maybritt" --run_id "maybritt_nicolai_zeros-constant_dec-1e-3_v2" "maybritt_nicolai_zeros-constant_dec-1e-3_temporal_v2" "maybritt_nicolai_zeros-constant_dec-1e-3_onehot" "maybritt_nicolai_zeros-constant_dec-1e-3_onehot_temporal" -1 -2 -3 -7 -8 -11 -12 -13 -15 -17 --project_specifier maybritt --variables tas pr sfcWind rsds --num_samples 9 --not_load_gan --not_load_benchmarks --not_load_diffusion --period 2090-2099

##############
#### extra metrics for plots per grid point, not included in the main evaluation
# eval per gridpoint, benchmarks
nice -n 7 python run_evals_script_metrics_per_gridpoint.py --save_name "full" --subfolder "benchmarks" --run_id "maybritt_3step_dense-conv_v3" --run_quantiles --run_multivariate --run_benchmarks --mode test_interpolation
# comparison for other run index (for model uncertainty)
nice -n 7 python run_evals_script_metrics_per_gridpoint.py --save_name "full" --subfolder "benchmarks" --run_id "maybritt_3step_dense-conv_v3" --run_quantiles --run_multivariate --run_benchmarks --mode test_interpolation --run_index 1
# eval per gridpoint, no benchmarks
nice -n 7 python run_evals_script_metrics_per_gridpoint.py --save_name "full" --subfolder "maybritt" --run_id "maybritt_nicolai_zeros-constant_dec-1e-3_v2" "maybritt_nicolai_zeros-constant_dec-1e-3_temporal_v2" "maybritt_nicolai_zeros-constant_dec-1e-3_onehot" "maybritt_nicolai_zeros-constant_dec-1e-3_onehot_temporal" --run_quantiles --run_multivariate --mode test_interpolation
