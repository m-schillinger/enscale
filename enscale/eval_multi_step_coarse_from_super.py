import torch
from modules import *
from modules_cnn import *
from modules_loc_variant import *
from utils import *
from data import *
import argparse
import pdb
import time

class args:
    # num_layers = [6, 6, 6, 6, 6]
    num_layers = [6, 2, 2, 2, 2]
    # hidden_dims = [200, 200, 200, 500, None]
    hidden_dims = [200, 12, 12, 12, 12]
    latent_dims = [None, 12, 12, 12, 12]
    preproc_layers = [True, False, False, False, False]
    preproc_dims = [50, None, None, None, None]
    noise_dims = [10, 5, 5, 5, 5]
    layer_shrinkages = [1, None, None, None, None]
    out_acts = [None, None, None, None, None]
    model_types = ["dense", "nicolai", "nicolai", "nicolai", "nicolai"]
    one_hot_options = [None, "argument", "argument", "argument", "argument"]
    one_hot_flags = [False, False, False, False, False]

    conv_dims = [None, None, None, None, None]
    kernel_sizes = [16, 8, 4, 2, 1]
    vars_as_channels = [False, False, False, False, False]

    hidden_dim_t = 200
    num_layer_t = 6
    preproc_layer_t = True
    preproc_dim_t = 50
    noise_dim_t = 10
    out_act_t = None
    layer_shrinkage_t = 1

    burn_ins = [399, 499, 99, 199]
    # specifications for nicolai layers
    num_neighbors_ups = [None, 9, 9, 9, 9]
    num_neighbors_res = [None, 25, 25, 25, 25]
    noise_dim_mlp = [None, 0, 0, 0, 0]
    batch_size = 128
    n_models = 1
    noise_std  = 1
    bn = False
    mlp = True
    val_dim = None
    variables = ["tas", "pr", "sfcWind", "rsds"]

    approx_unif = False # use approximate backtransformation for uniform for speed up    
    variables_lr =  ["tas", "pr", "sfcWind", "rsds", "psl"]
    n_visual = 5
    save_dir_super = None
    method = "eng_2step"
    avg_constraint = False # True for old runs, but then changed to False for newer version
    logit_transform = False
    sep_mean_std = False
    lambda_coarse = 1 #0.9 vs 1 (coarse from super vs pure coarse)
    norm_loss_batch = True
    norm_loss_loc = True
    server = "euler"
    save_quantiles = False #usually False, but set to true for saving quantiles over many samples for more accurate QL and MSE

    
def get_model(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader_in = get_data_2step_naive_avg(n_models=args.n_models,
                                                            variables=args.variables, variables_lr=args.variables_lr,
                                        batch_size=args.batch_size,
                                        norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                                        sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                                        kernel_size=args.kernel_sizes[0],
                                        logit=args.logit_transform,
                                        normal=args.normal_transform,
                                        server=args.server,)
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:args.n_visual].to(device), xc_tr_eval[:args.n_visual].to(device), y_tr_eval[:args.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:args.n_visual].to(device), xc_te_eval[:args.n_visual].to(device), y_te_eval[:args.n_visual].to(device)
    
    if args.variables_lr is not None:
        n_vars = len(args.variables_lr)
    else:
        n_vars = 5
    assert args.norm_method_output != "rank_val"
    in_dim = x_tr_eval.shape[1]
    
    #### get norm stats file
    norm_stats = {}
    for i in range(len(args.variables)):
        mode_unnorm = "hr"
        if args.variables[i] in ["pr", "sfcWind"] and args.sqrt_transform_out:
            name_str = "_sqrt"
        else:
            name_str = ""
        if args.norm_method_output == "normalise_pw":
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_pixelwise_" + args.variables[i] + "_train_ALL" + name_str + ".pt")
            norm_stats[args.variables[i]] = torch.load(ns_path, map_location=device)
        elif args.norm_method_output == "normalise_scalar":
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_full-data_" + args.variables[i] + "_train_ALL" + name_str + ".pt")
            norm_stats[args.variables[i]] = torch.load(ns_path, map_location=device)
        # TO DO: add norm stats for uniform_per_model, or maybe update this path
        elif args.norm_method_output == "uniform": #"hr_norm_stats_ecdf_matrix_" + data_type + "_train_" + "ALL" + name_str + ".pt")
            name_str = ""
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_ecdf_matrix_" + args.variables[i] + "_train_SUBSAMPLE" + name_str + ".pt")
            norm_stats[args.variables[i]] = torch.load(ns_path, map_location=device)
        else:
            norm_stats[args.variables[i]] = None
            
    in_dim = x_tr_eval.shape[1]
    
    if args.server == "euler":
        prefix = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/"
    elif args.server == "ada":
        prefix = "results/eng_2step/"
                   
    
    loaded_models = []
    for i in range(len(args.model_dirs)):
        num_layer = args.num_layers[i]
        hidden_dim = args.hidden_dims[i]
        preproc_layer = args.preproc_layers[i]
        noise_dim = args.noise_dims[i]
        layer_shrinkage = args.layer_shrinkages[i]
        out_act = args.out_acts[i]
        one_hot = args.one_hot_flags[i]
        burn_in = args.burn_ins[i]
        model_dir = args.model_dirs[i]

        save_dir = prefix + model_dir

        if args.model_types[i] == "dense":
            if i == 0:
                in_dim_model = in_dim
            else:
                in_dim_model = out_dim_model
                if args.one_hot_flags[i]:
                    in_dim_model += 7
            
            out_dim_model = int(128 / args.kernel_sizes[i])**2 * len(args.variables)
            print("parameters in model", in_dim_model, out_dim_model)
            print(num_layer, hidden_dim, noise_dim, preproc_layer, layer_shrinkage)
            model = StoUNet(
                in_dim_model,
                out_dim_model,
                num_layer,
                hidden_dim,
                noise_dim=noise_dim,
                add_bn=args.bn, out_act=out_act, 
                resblock=args.mlp, noise_std=args.noise_std,
                preproc_layer=args.preproc_layer_t, 
                input_dims_for_preproc=np.array(
                        [720  for k in range(n_vars)] +
                        [5, 7]),
                preproc_dim=args.preproc_dims[i],
                layer_shrinkage=layer_shrinkage,
            ).to(device)
            
        elif args.model_types[i] == "conv":
            # TO DO: make sure 4x resolution jump
            # TO DO: image size
             model = Generator4xExternalNoise(
                conv_dim=args.conv_dims[i],
                image_size=32,
                n_channels=len(args.variables),
                one_hot_channel=one_hot,
                one_hot_dim=7 if one_hot else None
            ).to(device)
    
        elif args.model_types[i] == "nicolai":
            assert i > 0
            print(in_dim_model, out_dim_model)
            
            if args.one_hot_in_super:
                num_classes = 8
            else:
                num_classes = 1
            model = RectUpsampleWithResiduals(int(128 / args.kernel_sizes[i-1]),
                            int(128 / args.kernel_sizes[i]),
                            n_features=len(args.variables),
                            num_classes=num_classes, 
                            num_classes_resid=1,
                            num_neighbors_ups=args.num_neighbors_ups[i],
                            num_neighbors_res=args.num_neighbors_res[i],
                            map_dim=args.latent_dims[i],
                            noise_dim=noise_dim,
                            mlp_hidden=hidden_dim,
                            mlp_depth=num_layer,
                            noise_dim_mlp=args.noise_dim_mlp[i],
                            double_linear=args.double_linear[i],
                            split_residuals=not args.not_split_residuals,
                            softmax=False
                            ).to(device)
    
        ckpt_path = os.path.join(save_dir, f"model_{burn_in}.pt")
        print(f"Loading model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        loaded_models.append(model)
    
    if not args.temporal:
        
        model_spec_list = \
            [
            ModelSpec(
                model = loaded_models[i],
                vars_as_channels = args.vars_as_channels[i],
                use_one_hot = args.one_hot_flags[i],
                noise_dim = args.noise_dims[i] * int(128 / args.kernel_sizes[i])**2, # latter part is the number of pixels for which we need noise
                one_hot_option = args.one_hot_options[i],
            )
            for i in range(len(loaded_models))
            ]
        sequential_model = HierarchicalWrapper(model_spec_list, n_vars=len(args.variables), one_hot_dim=7)
        coarse_model_marginal = loaded_models[0]
    else:
        # load coarse temporal model
        out_dim_model = int(128 / args.kernel_sizes[0])**2 * len(args.variables)
        model_temp = StoUNet(
                in_dim,
                out_dim_model,
                args.num_layer_t,
                args.hidden_dim_t,
                noise_dim=args.noise_dim_t,
                add_bn=args.bn, out_act=args.out_act_t, 
                resblock=args.mlp, noise_std=args.noise_std,
                preproc_layer=args.preproc_layer_t, 
                input_dims_for_preproc=np.array(
                                            [720  for k in range(n_vars)] +
                                            [5, 7] +
                                            [out_dim_model // len(args.variables) for k in range(len(args.variables))]),
                preproc_dim=args.preproc_dim_t,
                layer_shrinkage=args.layer_shrinkage_t
            ).to(device)
        save_dir_t = prefix + args.model_dir_t
        ckpt_path = os.path.join(save_dir_t, f"model_{args.burn_in_t}.pt")
        print(f"Loading model from {ckpt_path}")
        model_temp.load_state_dict(torch.load(ckpt_path, map_location=device))
        model_temp.eval()
        
        model_spec_list = \
            [ModelSpec(
                model = model_temp,
                vars_as_channels = False,
                use_one_hot = False, # because one hot doesn't need to be concatenated anymore
                noise_dim = args.noise_dim_t,
            )    
            ] + \
            [ # load remaining models
            ModelSpec(
                model = loaded_models[i],
                vars_as_channels = args.vars_as_channels[i],
                use_one_hot = args.one_hot_flags[i],
                noise_dim = args.noise_dims[i] * int(128 / args.kernel_sizes[i])**2,
                one_hot_option = args.one_hot_options[i],
            )
            for i in range(1, len(loaded_models))
            ]
        sequential_model = HierarchicalWrapper(model_spec_list, n_vars=len(args.variables), one_hot_dim=7)
        coarse_model_marginal = loaded_models[0]
    
    param_dict = {
        "save_dir": save_dir,
        "device": device,
        "norm_stats": norm_stats
    }
    if args.temporal:
        param_dict["save_dir_t"] = save_dir_t
        param_dict["burn_in_t"] = args.burn_in_t
        
    
    return sequential_model, coarse_model_marginal, param_dict

if __name__ == '__main__':
    # Load the model
    parser = argparse.ArgumentParser(description='Evaluate three-step coarse from super model')
    parser.add_argument('--temporal', action='store_true', help='Enable temporal mode')
    #counterfactuals = False #usually False, only for extra experiment True
    #split_coarse_super = False #usually False, only for extra experiment True
    #pure_super = False #usually False, only for extra experiment True
    parser.add_argument('--counterfactuals', action='store_true', help='Enable counterfactuals mode')
    parser.add_argument('--split_coarse_super', action='store_true', help='Enable split coarse super mode')
    parser.add_argument('--pure_super', action='store_true', help='Enable pure super mode')
    parser.add_argument('--version', type=str, default = "6", help='Version of the samples to save')
    parser.add_argument('--norm_option', type=str, choices = ["unif_norm", "pw"])
    parser.add_argument('--add_interm_loss', action='store_true', help='Enable intermediate loss')
    parser.add_argument('--add_mse_loss', action='store_true', help='Enable extra MSE loss')
    parser.add_argument('--use_double_linear', action='store_true')
    parser.add_argument('--nicolai_layers', action='store_true', help='Enable nicolai layers')
    parser.add_argument('--precip_zeros', type=str, default="random")
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay in optimisers')
    parser.add_argument('--one_hot_in_super',action='store_true')

    args_parsed = parser.parse_args()
    args.temporal = args_parsed.temporal
    args.counterfactuals = args_parsed.counterfactuals
    args.split_coarse_super = args_parsed.split_coarse_super
    args.pure_super = args_parsed.pure_super
    args.version = args_parsed.version
    args.add_interm_loss = args_parsed.add_interm_loss
    args.norm_option = args_parsed.norm_option
    args.nicolai_layers = args_parsed.nicolai_layers
    args.add_mse_loss = args_parsed.add_mse_loss
    args.use_double_linear = args_parsed.use_double_linear
    args.precip_zeros = args_parsed.precip_zeros
    args.weight_decay = args_parsed.weight_decay
    args.one_hot_in_super = args_parsed.one_hot_in_super

    if args.server == "ada":
        args.data_dir = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif args.server == "euler":
        args.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"

    if args.add_interm_loss:
        args.latent_dims = [None, 4, 4, 4, 4] # 4 if loss intermediate
    else:
        args.latent_dims = [None, 12, 12, 12, 12]

    if args.norm_option == "unif_norm":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = "uniform"
        args.normal_transform = True
        args.sqrt_transform_in = False
        args.sqrt_transform_out = False
    elif args.norm_option == "pw":
        args.norm_method_input = "normalise_scalar"
        args.norm_method_output = "normalise_pw"
        args.normal_transform = False
        args.sqrt_transform_in = True
        args.sqrt_transform_out = True

    if args.nicolai_layers:
        args.num_layers = [6, 2, 2, 2, 2]
        args.hidden_dims = [200, 12, 12, 12, 12]
        args.preproc_layers = [True, False, False, False, False]
        args.preproc_dims = [50, None, None, None, None]
        args.noise_dims = [10, 5, 5, 5, 5]
        args.layer_shrinkages = [1, None, None, None, None]        
        
        args.model_types = ["dense", "nicolai", "nicolai", "nicolai", "nicolai"]
        args.one_hot_options = [None, "argument", "argument", "argument", "argument"]
        args.conv_dims = [None, None, None, None, None]
        args.kernel_sizes = [16, 8, 4, 2, 1]
        args.vars_as_channels = [False, False, False, False, False]
        args.not_split_residuals = True
        
        if args.one_hot_in_super:
            args.one_hot_flags = [False, True, True, True, True]
        else:
            args.one_hot_flags = [False, False, False, False, False]
        
        # specifications for nicolai layers
        num_neighbors_ups = [None, 9, 9, 9, 9]
        num_neighbors_res = [None, 25, 25, 25, 25]
        noise_dim_mlp = [None, 0, 0, 0, 0]
        if args.use_double_linear:
            args.double_linear = [None, True, True, True, True]
        else:
            args.double_linear = [None, False, False, False, False]        
        
        if args.norm_option == "unif_norm" and args.precip_zeros == "constant" and args.weight_decay == 0:
            args.latent_dims = [None, 12, 12, 12, 12]
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosconstant/",
            ]
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/"
            args.burn_in_t = 299
            
        elif args.norm_option == "unif_norm" and args.precip_zeros == "random" and args.weight_decay == 0:
            args.latent_dims = [None, 12, 12, 12, 12]
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosrandom/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec0_lam-mse0_split-residFalse_zerosrandom/",
            ]
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosrandom/"
            args.burn_in_t = 299
            
        elif args.norm_option == "unif_norm" and args.precip_zeros == "constant" and args.weight_decay == 1e-3 and not args.one_hot_in_super:
            args.latent_dims = [None, 12, 12, 12, 12]
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant/",
            ]
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/"
            args.burn_in_t = 299
            
        # with one hot
        elif args.norm_option == "unif_norm" and args.precip_zeros == "constant" and args.weight_decay == 1e-3 and args.one_hot_in_super:
            args.latent_dims = [None, 12, 12, 12, 12]
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant_onehot-simple/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant_onehot-simple/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant_onehot-simple/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosconstant_onehot-simple/",
            ]
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosconstant/"
            args.burn_in_t = 299
            
        elif args.norm_option == "unif_norm" and args.precip_zeros == "random" and args.weight_decay == 1e-3: 
            args.latent_dims = [None, 12, 12, 12, 12]
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosrandom/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform__normal-transform_dec1e-3_lam-mse0_split-residFalse_zerosrandom/",
            ]
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosrandom/"
            args.burn_in_t = 299
            
        elif args.norm_option == "unif_norm" and args.add_interm_loss:
            args.latent_dims = [None, 4, 4, 4, 4]  # 4 if loss intermediate
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_norm-loss-per-var-p4_euler/",
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform_dec-0.0_normal-transform_loss-on-intermediate_no-split-residuals_pretrain-mse/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform_dec-0.0_normal-transform_loss-on-intermediate_no-split-residuals_pretrain-mse/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform_dec-0.0_normal-transform_loss-on-intermediate_no-split-residuals_pretrain-mse/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-uniform_dec-0.0_normal-transform_loss-on-intermediate_no-split-residuals_pretrain-mse/",
            ]
            #args.burn_ins = [399, 199, 199, 199, 99]        
            args.burn_ins = [199, 199, 199, 199, 99]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-uniform_normal_preproc_zerosrandom/"
            args.burn_in_t = 299
            
        elif args.norm_option == "pw" and args.use_double_linear:
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.model_dirs = [
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_double-linear_no-split-residuals_pretrain-mse/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_double-linear_no-split-residuals_pretrain-mse/",   
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_double-linear_no-split-residuals_pretrain-mse/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_double-linear_no-split-residuals_pretrain-mse/",
            ]
            args.burn_ins = [399, 499, 499, 499, 199]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_norm-loss-per-var-p4"
            args.burn_in_t = 299
            
        elif args.norm_option == "pw" and not args.use_double_linear:
            args.latent_dims = [None, 12, 12, 12, 12] 
            args.model_dirs = [
                #"coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc_norm-loss-per-var-p4/",
                "coarse/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_preproc/",
                "super/lr16_hr8/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_no-split-residuals_pretrain-mse/",
                "super/lr8_hr4/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_no-split-residuals_pretrain-mse/",
                "super/lr4_hr2/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_no-split-residuals_pretrain-mse/",
                "super/lr2_hr1/var-tas_pr_sfcWind_rsds/loc-specific-layers_norm-out-normalise_pw_dec-0.0_no-split-residuals_pretrain-mse/",
            ]
            args.burn_ins = [199, 499, 499, 499, 499]
            
            args.model_dir_t = "coarse_temporal/var-tas_pr_sfcWind_rsds/hd-200_num-lay-6_norm-out-normalise_pw_norm-loss-per-var-p4"
            args.burn_in_t = 299
            
    model, coarse_model_marginal, param_dict = get_model(args)
    
    device = param_dict["device"]
    norm_stats = param_dict["norm_stats"]
    
    if args.temporal:
        save_dir_t = param_dict["save_dir_t"]
        burn_in_t = param_dict["burn_in_t"]


    if args.server == "euler":
        save_dir_samples = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/samples_multivariate/" + f"maybritt_{args.version}/"
    else:
        save_dir_samples = "/r/scratch/groups/nm/downscaling/samples_multivariate/" + f"maybritt_{args.version}/"   
    
    print("saving samples in ", save_dir_samples)
    os.makedirs(save_dir_samples, exist_ok=True)

    with open(os.path.join(save_dir_samples, "ckpt_dirs.txt"), "w") as f:
        for i in range(len(args.model_dirs)):
            f.write(f"Model {i} checkpoint: {os.path.join(args.model_dirs[i], f'model_{args.burn_ins[i]}.pt')}\n")
    
        if args.temporal:
            f.write(f"Saving temporal model: {os.path.join(args.model_dir_t, f'model_{args.burn_in_t}.pt')}\n")
    
    with open(os.path.join(save_dir_samples, "model_size.txt"), "w") as f:
        n_params = count_parameters_wrapper(model)
        f.write(f"Total parameters in model: {n_params}\n")
    
    # run_indices = np.arange(0,8)
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(args.data_dir)
    gcm_indices = torch.tensor([gcm_dict[gcm] for gcm in gcm_list])
    one_hot_gcm = torch.nn.functional.one_hot(gcm_indices)
    rcm_indices = torch.tensor([rcm_dict[rcm] for rcm in rcm_list])
    one_hot_rcm = torch.nn.functional.one_hot(rcm_indices)
    mode_unnorm = "hr"
    counterfactuals = args.counterfactuals
    one_hot_dim = 7
    if not counterfactuals:
        k_range = np.arange(0, 8)
    else:
        k_range = np.arange(0, 1)
        
        i_new = 2
            
    for k in k_range:
        print(f"Index {k}")
        if args.split_coarse_super or args.pure_super:
            modes = ["test_interpolation"]
        else:
            modes = ["test_interpolation", "test_extrapolation"]
        for mode in modes:
            print(f"Mode {mode}")
            test_loader_in, _ = get_data_2step_naive_avg(run_indices=[k], mode = mode,
                                                        variables=args.variables, variables_lr=args.variables_lr,
                                                        batch_size=args.batch_size, shuffle=False,
                                                        norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                                                        sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                                                        kernel_size=args.kernel_sizes[0],
                                                        logit=args.logit_transform,
                                                        normal=args.normal_transform,
                                                        return_timepair=args.temporal,
                                                        server=args.server,
                                                        precip_zeros=args.precip_zeros)
            samples = []
            if counterfactuals:
                samples_counterfact = []
            model.eval()
            model.to(device)
            
            start = time.time()
            
            for idx, data_batch in enumerate(test_loader_in):
                if not args.temporal:
                    x, xc, y = data_batch
                    x, xc, y = x.to(device), xc.to(device), y.to(device)
                else:
                    x_prev, xc_prev, y_prev, x, xc, y = data_batch
                    x_prev, xc_prev, y_prev, x, xc, y = x_prev.to(device), xc_prev.to(device), y_prev.to(device), x.to(device), xc.to(device), y.to(device)
                
                if args.split_coarse_super:
                    # avoid too large data
                    if idx > 2:
                        break
                    gen_coarse_list = []
                elif args.pure_super:
                    gen_coarse_list = []
                
                with torch.no_grad():
                    cls_ids = get_run_index_from_onehot(x[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
                    if args.temporal:
                        start_xc = coarse_model_marginal.sample(x_prev[:1, ...], sample_size=1).to(device)
                        # start_xc = model.coarse_model.sample(x_prev[:1, ...], sample_size=1).to(device)
                        gen = model.sample_temporal(x, sample_size=9, start_xc=start_xc[0, :, 0], x_onehot=x, cls_ids=cls_ids).to(device)
                        gen = gen.view(x.shape[0], len(args.variables), -1, 9)
                    elif not counterfactuals and not args.split_coarse_super and not args.pure_super and not args.save_quantiles:
                        gen = model.sample(x.to(device), sample_size=9, x_onehot=x, cls_ids=cls_ids).to(device)
                        gen = gen.view(x.shape[0], len(args.variables), -1, 9)
                        
                    # also alternative one hot experiment
                    elif counterfactuals:
                        gen_list = []
                        gen_counterfact_list = []
                        x_counterfact = x.clone()
                        x_counterfact[:, -one_hot_dim:] = torch.cat([one_hot_gcm[i_new], one_hot_rcm[i_new]]).repeat(x.shape[0], 1)
                        cls_ids_counterfact = get_run_index_from_onehot(x_counterfact[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
                        npix_list = np.array([int(128 / args.kernel_sizes[i])**2 for i in range(len(args.kernel_sizes))])
                        noise_dim_super = sum(args.noise_dims[1:] * npix_list[1:])
                        for j in range(9):
                            # need separate generations coarse and super, to ensure noise is the same for both
                            eps_coarse = torch.randn(x.shape[0], args.noise_dims[0] * (args.num_layers[0] // 2) * 2, device=device)

                            # in new hierarchical wrapper, there is no coarse model anymore; therefore switch to coarse_model_marginal
                            # x_rcmc = model.coarse_model(x, eps=eps_coarse) 
                            # x_rcmc_counterfact = model.coarse_model(x_counterfact, eps=eps_coarse)
                            x_rcmc = coarse_model_marginal(x, eps=eps_coarse)
                            x_rcmc_counterfact = coarse_model_marginal(x_counterfact, eps=eps_coarse)
                                                        
                            eps_super = torch.randn(x.shape[0], noise_dim_super, device=device)
                            gen =  model._apply_remaining_models(x_rcmc, eps=eps_super, x_onehot=x, cls_ids=cls_ids, start_idx=1, return_intermediates=False)
                            gen_counterfact = model._apply_remaining_models(x_rcmc_counterfact, eps=eps_super, x_onehot=x_counterfact, cls_ids=cls_ids_counterfact, start_idx=1, return_intermediates=False)
                            #gen = model.super_model(x_rcmc, eps=eps_super)
                            #gen_counterfact = model.super_model(x_rcmc_counterfact, eps=eps_super)
                            
                            gen = gen.view(gen.shape[0], len(args.variables), -1)
                            gen_counterfact = gen_counterfact.view(gen_counterfact.shape[0], len(args.variables), -1)
                            gen_list.append(gen)
                            gen_counterfact_list.append(gen_counterfact)
                        
                        print(gen_list[0].shape)
                        gen = torch.stack(gen_list, dim=-1)
                        gen_counterfact = torch.stack(gen_counterfact_list, dim=-1)
                        
                        
                    elif args.split_coarse_super:
                        # sample first from coarse model and then for each sample, sample again from super model
                        
                        # OLD x_rcmc = model.coarse_model.sample(x.to(device), sample_size=9).to(device)
                        x_rcmc = coarse_model_marginal.sample(x.to(device), sample_size=9).to(device)
                        
                        gen_super_list = []
                        for i in range(9):
                            for j in range(9):
                                gen_super = model._apply_remaining_models(
                                    x_rcmc[..., i], 
                                    x_onehot=x, 
                                    cls_ids=cls_ids, 
                                    start_idx=1, 
                                    return_intermediates=False
                                )
                                gen_super = gen_super.view(x.shape[0], len(args.variables), -1)
                                gen_super_list.append(gen_super)
                        gen = torch.stack(gen_super_list, dim=-1)
                        gen_coarse_list.append(x_rcmc)
                    
                    elif args.pure_super:
                        # to do: shape of xc? If not conv_super_coarse, then need to flatten
                        xc = xc.view(xc.shape[0], -1)
                        gen = torch.stack([
                            model._apply_remaining_models(
                                    xc, 
                                    x_onehot=x, 
                                    cls_ids=cls_ids, 
                                    start_idx=1, 
                                    return_intermediates=False
                                ).view(xc.shape[0], len(args.variables), -1)
                        for j in range(9)], dim=-1)
                        # gen = model.super_model.sample(xc, sample_size=9, x_onehot=x.to(device)).to(device)
                        gen_coarse_list.append(xc)   
                        
                    elif args.save_quantiles:
                        gen = model.sample(x.to(device), sample_size=100).to(device) 
                samples.append(gen.detach().cpu())
                if counterfactuals:
                    samples_counterfact.append(gen_counterfact.detach().cpu())
                
            end = time.time()
            print("Time taken for sampling: ", end - start)
            
            samples_norm = torch.cat(samples)
            if counterfactuals:
                samples_counterfact_norm = torch.cat(samples_counterfact)
            
            start = time.time()
            # do normalisation here with larger batch size 
            batch_size_unnorm = 8192 
            n_batches = np.ceil(samples_norm.shape[0] / batch_size_unnorm)
            print(n_batches)
            samples_raw = []
            samples_counterfact_raw = []
            for i in range(int(n_batches)):
                gen = samples_norm[i * batch_size_unnorm: (i+1) * batch_size_unnorm]
                if counterfactuals:
                    gen_counterfact = samples_counterfact_norm[i * batch_size_unnorm: (i+1) * batch_size_unnorm]
                
                gen_raw_allvars_list = []
                if counterfactuals:
                    gen_raw_counterfact_allvars_list = []
                for i in range(len(args.variables)):
                    gen_raw_var_list = []
                    
                    if counterfactuals:
                        gen_raw_counterfact_var_list = []
                    for j in range(gen.shape[-1]):
                        gen_raw = unnormalise(gen[:, i, :, j], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                            logit=args.logit_transform, 
                                            normal=args.normal_transform,
                                            approx_unif=args.approx_unif, n_keep_vals=1000, interp_step=1)
                        gen_raw_var_list.append(gen_raw)
                        
                        if counterfactuals:
                            gen_raw_counterfact = unnormalise(gen_counterfact[:, i, :, j], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                            logit=args.logit_transform, 
                                            normal=args.normal_transform,
                                            approx_unif=args.approx_unif, n_keep_vals=1000, interp_step=1)
                            gen_raw_counterfact_var_list.append(gen_raw_counterfact)
                        
                    gen_raw_var = torch.stack(gen_raw_var_list, dim=-1)
                    gen_raw_allvars_list.append(gen_raw_var)
                    
                    if counterfactuals:
                        gen_raw_counterfact_var = torch.stack(gen_raw_counterfact_var_list, dim=-1)
                        gen_raw_counterfact_allvars_list.append(gen_raw_counterfact_var)
                        
                gen_raw_allvars = torch.stack(gen_raw_allvars_list, dim=1)
                
                if counterfactuals:
                    gen_raw_counterfact_allvars = torch.stack(gen_raw_counterfact_allvars_list, dim=1)
                    
                # samples.append(gen_raw_allvars)
                samples_raw.append(gen_raw_allvars)
                if counterfactuals:
                #    samples_counterfact.append(gen_raw_counterfact_allvars)
                    samples_counterfact_raw.append(gen_raw_counterfact_allvars)
                    
            samples_raw = torch.cat(samples_raw, dim=0)
            if counterfactuals:
                samples_counterfact_raw = torch.cat(samples_counterfact_raw, dim=0)
            
            end = time.time()
            print("Time taken for unnormalisation: ", end - start)
            
            suffix = "" if not args.approx_unif else "_approx"
            if args.split_coarse_super:
                gen_coarse = torch.cat(gen_coarse_list)
                torch.save(gen_coarse, save_dir_samples + f'idx{k}_inter_gen-coarse{suffix}.pt')
                torch.save(samples_raw, save_dir_samples + f'idx{k}_inter_gen-super-from-coarse{suffix}.pt')
            elif args.pure_super:
                gen_coarse = torch.cat(gen_coarse_list)
                torch.save(gen_coarse, save_dir_samples + f'idx{k}_inter_true-coarse{suffix}.pt')
                torch.save(samples_raw, save_dir_samples + f'idx{k}_inter_gen-pure-super{suffix}.pt')
            
            elif not counterfactuals and not args.save_quantiles:
                if mode == "test_interpolation":
                    torch.save(samples_norm, save_dir_samples + f'idx{k}_inter{suffix}_unif.pt')
                    torch.save(samples_raw, save_dir_samples + f'idx{k}_inter{suffix}.pt')
                elif mode == "test_extrapolation":
                    torch.save(samples_norm, save_dir_samples + f'idx{k}_extra{suffix}_unif.pt')
                    torch.save(samples_raw, save_dir_samples + f'idx{k}_extra{suffix}.pt')
            
            elif counterfactuals:
                if mode == "test_interpolation":
                    torch.save(samples_raw, save_dir_samples + f'idx{k}_inter_counterfact_base-for{i_new}{suffix}.pt')
                elif mode == "test_extrapolation":
                    torch.save(samples_raw, save_dir_samples + f'idx{k}_extra_counterfact_base-for{i_new}{suffix}.pt')
                if mode == "test_interpolation":
                    torch.save(samples_counterfact_raw, save_dir_samples + f'idx{k}_inter_counterfact{i_new}{suffix}.pt')
                elif mode == "test_extrapolation":
                    torch.save(samples_counterfact_raw, save_dir_samples + f'idx{k}_extra_counterfact{i_new}{suffix}.pt')
                    
            elif args.save_quantiles:
                # get quantiles for each location
                qs = [0.1, 0.5, 0.9]
                for q in qs:
                    q_samples = torch.quantile(samples_raw, q, dim=-1)
                    if mode == "test_interpolation":
                        torch.save(q_samples, save_dir_samples + f'idx{k}_inter_quantile-pw{q}{suffix}.pt')
                    elif mode == "test_extrapolation":
                        torch.save(q_samples, save_dir_samples + f'idx{k}_extra_quantile-pw{q}{suffix}.pt')
                    
                    q_sp_mean = torch.quantile(samples_raw.mean(dims=(-2, -3)), q, dim = -1)
                    # torch.save(q_sp_mean, save_dir_samples + f'idx{k}_quantile-sp-mean{q}.pt')
                    if mode == "test_interpolation":
                        torch.save(q_sp_mean, save_dir_samples + f'idx{k}_inter_quantile-sp-mean{q}{suffix}.pt')
                    elif mode == "test_extrapolation":
                        torch.save(q_sp_mean, save_dir_samples + f'idx{k}_extra_quantile-sp-mean{q}{suffix}.pt')
                    
                    q_sp_max = torch.quantile(samples_raw.max(dims=(-2, -3)), q, dim = -1)
                    # torch.save(q_sp_max, save_dir_samples + f'idx{k}_quantile-sp-max{q}.pt')
                    if mode == "test_interpolation":
                        torch.save(q_sp_max, save_dir_samples + f'idx{k}_inter_quantile-sp-max{q}{suffix}.pt')
                    elif mode == "test_extrapolation":
                        torch.save(q_sp_max, save_dir_samples + f'idx{k}_extra_quantile-sp-max{q}{suffix}.pt')
                    
                q_cond_mean = samples_raw.mean(dims=-1)
                # torch.save(q_sp_mean, save_dir_samples + f'idx{k}_cond-mean.pt')
                if mode == "test_interpolation":
                    torch.save(q_cond_mean, save_dir_samples + f'idx{k}_inter_cond-mean{suffix}.pt')
                elif mode == "test_extrapolation":
                    torch.save(q_cond_mean, save_dir_samples + f'idx{k}_extra_cond-mean{suffix}.pt')