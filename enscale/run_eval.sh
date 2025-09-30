# basic sample generation
python eval_multi_step_coarse_from_super.py --norm_option unif_norm --nicolai_layers --precip_zeros constant --weight_decay 1e-3 --version 'nicolai_zeros-constant_dec-1e-3_onehot' --one_hot_in_super
python eval_multi_step_coarse_from_super.py --norm_option unif_norm --nicolai_layers --precip_zeros constant --weight_decay 1e-3 --version 'nicolai_zeros-constant_dec-1e-3_onehot_temporal' --temporal --one_hot_in_super

#####
# special sample generation for special plots / variability analysis
# split coarse from super
python eval_multi_step_coarse_from_super.py --norm_option unif_norm --nicolai_layers --precip_zeros constant --weight_decay 1e-3 --version 'nicolai_zeros-constant_dec-1e-3_onehot' --one_hot_in_super --split_coarse_super

# counterfactuals
python eval_multi_step_coarse_from_super.py --norm_option unif_norm --nicolai_layers --precip_zeros constant --weight_decay 1e-3 --version 'nicolai_zeros-constant_dec-1e-3_onehot' --one_hot_in_super --counterfactuals
# pure super
python eval_multi_step_coarse_from_super.py --norm_option unif_norm --nicolai_layers --precip_zeros constant --weight_decay 1e-3 --version 'nicolai_zeros-constant_dec-1e-3_onehot' --one_hot_in_super --pure_super
