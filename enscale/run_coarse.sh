# marginal coarse model
python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_normal_preproc_zerosconstant' --precip_zeros constant --method eng_2step --variables tas pr sfcWind rsds \
 --variables_lr tas pr sfcWind rsds psl --num_epochs 500 --norm_method_input normalise_scalar --norm_method_output uniform --normal_transform --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
 --agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 50 --save_model_every 50

# temporal coarse model
python train_only-coarse.py --num_layer 6 --hidden_dim 200 --out_act None --save_name '_normal_preproc_zerosconstant' --precip_zeros constant --method eng_temporal --variables tas pr sfcWind rsds \
--variables_lr tas pr sfcWind rsds psl --num_epochs 500 --norm_method_input normalise_scalar --norm_method_output uniform --normal_transform --kernel_size_lr 16 --layer_shrinkage 1 --noise_dim 10 \
--agg_norm_loss mean --lambda_norm_loss_loc 1  --p_norm_loss_loc 4 --p_norm_loss_batch 4 --preproc_layer --preproc_dim 50 --save_model_every 50