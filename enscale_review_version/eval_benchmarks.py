import torch
from modules import *
from utils import *
from data import *


class args:
    num_layer = 6
    num_layer_enc = None
    hidden_dim = 500
    noise_dim = 0
    batch_size = 512
    n_models = 1
    noise_std  = 1
    bn = False
    mlp = True
    val_dim = None
    out_act = None
    preproc_layer = False
    data_dir = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    burn_in = 24
    variables =  ['tas', 'pr', 'sfcWind', 'rsds']
    variables_lr = ['tas', 'pr', 'sfcWind', 'rsds', 'psl']
    sqrt_transform_in = False
    sqrt_transform_out = False
    norm_method_input = "normalise_scalar"
    norm_method_output = "uniform"
    kernel_size_lr = 16
    layer_shrinkage = 16
    n_visual = 5
    save_dir_super = None
    method = "nn_det_per_variable"
    avg_constraint = False
    preproc_dim = 100
    logit_transform = True
    sep_mean_std = False
    conv = False
    server = "euler"
    clip_quantile_data = False
    tr_te_split = "random"
    test_model_index=None
    train_model_index=None
    
if __name__ == "__main__":
    device = torch.device('cuda')
    train_loader, test_loader_in = get_data_2step_naive_avg(n_models=args.n_models, variables=args.variables, 
                                                            variables_lr=args.variables_lr,
                                            batch_size=args.batch_size,
                                            norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                                            sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                                            kernel_size=args.kernel_size_lr, clip_quantile=args.clip_quantile_data,
                                            tr_te_split=args.tr_te_split, 
                                            test_model_index=args.test_model_index,
                                            train_model_index=args.train_model_index,
                                            logit=args.logit_transform,
                                            server = args.server)
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:args.n_visual].to(device), xc_tr_eval[:args.n_visual].to(device), y_tr_eval[:args.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:args.n_visual].to(device), xc_te_eval[:args.n_visual].to(device), y_te_eval[:args.n_visual].to(device)
    
    if args.server == "euler":
        args.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    
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
        elif args.norm_method_output == "uniform": #"hr_norm_stats_ecdf_matrix_" + data_type + "_train_" + "ALL" + name_str + ".pt")
            name_str = ""
            ns_path = os.path.join(args.data_dir, "norm_stats", f"{mode_unnorm}_norm_stats_ecdf_matrix_" + args.variables[i] + "_train_SUBSAMPLE" + name_str + ".pt")
            norm_stats[args.variables[i]] = torch.load(ns_path, map_location=device)
        else:
            norm_stats[args.variables[i]] = None
            
    if args.method == "nn_det_per_variable" or args.method == "crps_pw_per_variable" or args.method == "eng_unet_per_variable":
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        super_model_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'num_layer': args.num_layer,
            'hidden_dim': args.hidden_dim,
            'noise_dim': args.noise_dim,
            'add_bn': args.bn,
            'out_act': args.out_act,
            'resblock': args.mlp,
            'noise_std': args.noise_std,
            'preproc_layer': args.preproc_layer,
            'n_vars': 1,
            'layer_shrinkage': args.layer_shrinkage,
        }
        model = MultivariateStoUNetWrapper(len(args.variables), super_model_params, expand_variables=False, split_input=False).to(device)    
    
    save_dir = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/nn_det_per_variable/var-tas_pr_sfcWind_rsds/hidden500_norm-in-normalise_scalar_norm-out-uniform/"
    ckpt_dir = save_dir + f"model_{args.burn_in}.pt"
    model.load_state_dict(torch.load(ckpt_dir, map_location=device))
    
    save_dir_samples = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/nn_det_per_variable/burnin24_v1/"
    os.makedirs(save_dir_samples, exist_ok=True)
    
    for k in range(8):
        print(k)
        for mode in ["test_interpolation", "test_extrapolation"]:
            print(mode)
            test_loader_in, _ = get_data_2step_naive_avg(run_indices=[k], mode = mode,
                                                         test_size=0.0,
                                                        shuffle=False, #n_models=1,
                                                        variables=args.variables, variables_lr=args.variables_lr,
                                                        batch_size=args.batch_size,
                                                        norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                                                        sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                                                        kernel_size=args.kernel_size_lr,
                                                        logit=args.logit_transform,
                                                        server = args.server)

            trues = {}
            samples = {}
            for var in args.variables:
                trues[var] = []
                samples[var] = []
            model.eval()
            for idx, data_batch in enumerate(test_loader_in):
                x, xc, y = data_batch
                x, xc, y = x.to(device), xc.to(device), y.to(device)
                with torch.no_grad():
                    gen = model.sample(x.to(device), sample_size=1).to(device)
                    # y = y.view(y.shape[0], -1)
                    for i, var in enumerate(args.variables): 
                        if norm_stats is not None:
                            norm_stats_var = norm_stats[args.variables[i]]
                        else:
                            norm_stats_var = None
                        norm_method = args.norm_method_output
                        sqrt_transform = args.sqrt_transform_out
                        logit = args.logit_transform
                        # y_var = unnormalise(y[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, 
                        #                    norm_method=norm_method, norm_stats=norm_stats_var, logit=logit, final_square=False).unsqueeze(1)
                        gen1_var = unnormalise(gen[:, i*128*128:(i+1)*128*128, 0], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, 
                                            norm_method=norm_method, norm_stats=norm_stats_var, sep_mean_std=args.sep_mean_std, logit=logit, final_square=False).unsqueeze(1)
                        # trues[var].append(y_var.cpu())
                        samples[var].append(gen1_var.cpu())
            for var in args.variables:
                samples[var] = torch.cat(samples[var], dim = 0)
                # trues[var] = torch.cat(trues[var], dim = 0)
                if mode == "train":
                    # torch.save(samples[var], save_dir_samples + f'{var}/idx{k}_inter.pt')
                    torch.save(samples[var], save_dir_samples + f'{var}_idx{k}_train.pt')
                elif mode == "test_interpolation":
                    torch.save(samples[var], save_dir_samples + f'{var}_idx{k}_inter.pt')
                elif mode == "test_extrapolation":
                    torch.save(samples[var], save_dir_samples + f'{var}_idx{k}_extra.pt')
            model.train()   
                