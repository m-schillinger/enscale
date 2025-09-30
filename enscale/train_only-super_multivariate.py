import torch
import os
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modules import StoUNet, RankValModel, LinearModel, GCMCoarseRCMModel, MLPConvWrapper
from modules_cnn import Generator16x, Generator4x, Generator4xConcat, Generator2x
from modules_loc_variant import RectUpsampleWithResiduals, RectUpsampler
from loss_func import energy_loss_two_sample, avg_constraint_per_var, energy_loss_multivariate_summed, norm_loss_multivariate_summed
import torch.nn.functional as F
import torch.linalg as LA

from data import get_data, get_data_2step, get_data_2step_naive_avg
from config import get_config
from utils import *
import sys
import pdb
import time
sys.path.append("..")


def visual_sample(model, x, y, save_dir, norm_method=None, norm_stats=None, sqrt_transform=True, square_data=False, mode_unnorm = "hr", 
                  one_hot_dim=None, logit=False, normal=False, fft=False, one_hot_in_super=False, conv=False, x_one_hot=None):
    if not args.dropout:
        model.eval()
    with torch.no_grad():
        if not args.conv and not args.conv_concat:
            x = x.view(x.shape[0], -1)
            y = y.view(y.shape[0], -1)
        if args.add_x_in_super: # DIRTY
            x = torch.cat([x, x_one_hot[:, :720]], dim=1)
        
        if not args.nicolai_layers:
            x = add_one_hot(x, one_hot_in_super=one_hot_in_super, conv=conv, x=x_one_hot, one_hot_dim=one_hot_dim)
        
        if args.latent_dim == len(args.variables) and args.nicolai_layers and args.add_intermediate_loss:
            plot_intermediate = True
            if args.double_linear:
                gen, _, y_interm = model(x, return_latent=True)
                gen2, _, y_interm2 = model(x, return_latent=True)
            else:
                gen, y_interm = model(x, return_latent=True)
                gen2, y_interm2 = model(x, return_latent=True)
        elif args.nicolai_layers and one_hot_in_super:
            cls_ids = get_run_index_from_onehot(x_one_hot[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
            gen, y_upsampled = model(x, cls_ids=cls_ids, return_mean=True)
            gen2, y_upsampled2 = model(x, cls_ids=cls_ids, return_mean=True)
            y_interm = gen - y_upsampled.view(gen.shape[0], -1)
            y_interm2 = gen2 - y_upsampled2.view(gen.shape[0], -1)
            plot_intermediate = True
        elif args.nicolai_layers:
            gen, y_upsampled = model(x, return_mean=True)
            gen2, y_upsampled2 = model(x, return_mean=True)
            y_interm = gen - y_upsampled.view(gen.shape[0], -1)
            y_interm2 = gen2 - y_upsampled2.view(gen.shape[0], -1)
            plot_intermediate = True
        else:
            plot_intermediate = False
            gen = model(x)
            gen2 = model(x)    
    for i in range(len(args.variables)):
        if len(gen.shape) == 3:
            gen_var = gen[:, i, :]
            gen_var2 = gen2[:, i, :]
        else:
            dim_per_var = gen.size(1) // len(args.variables)
            gen_var = gen[:, i * dim_per_var:(i + 1) * dim_per_var]
            gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
            
        if len(y.shape) == 3:
            y_var = y[:, i, :]
        else:
            dim_per_var = gen.size(1) // len(args.variables)
            y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
        
        if norm_stats is not None:
            norm_stats_var = norm_stats[args.variables[i]]
        else:
            norm_stats_var = None
        gen_var = unnormalise(gen_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var, 
                    final_square=square_data, logit=logit, normal=normal).unsqueeze(1)
        gen_var2 = unnormalise(gen_var2, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var,
                    final_square=square_data, logit=logit, normal=normal).unsqueeze(1) 
        y_var = unnormalise(y_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var, 
                           final_square=square_data, logit=logit, normal=normal).unsqueeze(1)
        
        if plot_intermediate and norm_stats_var is None:
            # shape of intermediate (B, npix, map_dim) 
            intermediate_var = y_interm[:, i * dim_per_var:(i + 1) * dim_per_var].view(y_interm.shape[0], 1, 128 // args.kernel_size_hr, 128 // args.kernel_size_hr)
            intermediate_var2 = y_interm2[:, i * dim_per_var:(i + 1) * dim_per_var].view(y_interm2.shape[0], 1, 128 // args.kernel_size_hr, 128 // args.kernel_size_hr)
            sample = torch.cat([y_var.cpu(), intermediate_var.cpu(), intermediate_var2.cpu(), gen_var.cpu(), gen_var2.cpu() ])
        else:
            sample = torch.cat([y_var.cpu(), gen_var.cpu(), gen_var2.cpu()])
        sample = torch.clamp(sample, torch.quantile(y_var, 0.0005).item(), torch.quantile(y_var, 0.9995).item())
        plt.matshow(make_grid(sample, nrow=y.shape[0]).permute(1, 2, 0)[:,:,0], cmap="rainbow"); plt.axis('off'); 
        plt.savefig(save_dir + f"_var-{args.variables[i]}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
        # save_image(sample, save_dir, normalize=True, scale_each=True)
    model.train()
    

def get_eval_samples(current_model, current_test_loader, mode_unnorm="hr", norm_method=None, norm_stats=None, 
                     input_mode = "x", output_mode = "y", logit=False, normal=False, one_hot_in_super=False, conv=False, one_hot_dim=None):
    if not args.dropout:
        current_model.eval()
    samples = []
    trues = []
    with torch.no_grad():
        n_batches = 0
        for data_te in current_test_loader:
            x_te, xc_te, y_te = data_te
            x_te, xc_te, y_te = x_te.to(device), xc_te.to(device), y_te.to(device)
            x_te = x_te.view(x_te.shape[0], -1)
            
            if not args.conv and not args.conv_concat:
                # x_te = x_te.view(x_te.shape[0], -1)
                y_te = y_te.view(y_te.shape[0], -1)
                xc_te = xc_te.view(xc_te.shape[0], -1)
            
            if input_mode == "x":
                gen = current_model.sample(x_te, sample_size=5)
            elif input_mode == "xc":
                if args.add_x_in_super:
                    xc_te = torch.cat([xc_te, x_te[:, :720]], dim=1)
                    
                if not args.nicolai_layers:
                    xc_te = add_one_hot(xc_te, one_hot_in_super=one_hot_in_super, conv=conv, x=x_te, one_hot_dim=one_hot_dim)
                    gen = current_model.sample(xc_te, sample_size=5)
                elif one_hot_in_super:
                    cls_ids = get_run_index_from_onehot(x_te[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
                    gen = current_model.sample(xc_te, cls_ids=cls_ids, sample_size=5)
                else:
                    gen = current_model.sample(xc_te, sample_size=5)
            try:                
                gen_raw_allvars_list = []
                for i in range(len(args.variables)):
                    
                    if norm_stats is not None:
                        norm_stats_var = norm_stats[args.variables[i]]
                    else:
                        norm_stats_var = None
                    
                    gen_raw_var_list = []
                    for j in range(5):
                        if len(gen.shape) == 4: # 
                            gen_var_sample = gen[:, i, :, j]
                        elif len(gen.shape) == 3:
                            dim_per_var = gen.size(1) // len(args.variables)
                            gen_var_sample = gen[:, i * dim_per_var:(i + 1) * dim_per_var, j]
                        gen_raw = unnormalise(gen_var_sample, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=norm_method, norm_stats=norm_stats_var, sep_mean_std=args.sep_mean_std,
                                            logit=logit, normal=normal)
                        gen_raw_var_list.append(gen_raw)
                        
                    gen_raw_var = torch.stack(gen_raw_var_list, dim=-1)
                    gen_raw_allvars_list.append(gen_raw_var)
                gen_raw_allvars = torch.stack(gen_raw_allvars_list, dim=1)        
                
            except RuntimeError:
                pdb.set_trace()

            if output_mode == "y":
                y = y_te
            elif output_mode == "xc":
                y = xc_te
                
            try:
                y_te_raw_allvars_list = []
                for i in range(len(args.variables)):
                    
                    if norm_stats is not None:
                        norm_stats_var = norm_stats[args.variables[i]]
                    else:
                        norm_stats_var = None
                    
                    if len(y.shape) == 3:
                        y_var = y[:, i, :]
                    elif len(y.shape) == 2:
                        dim_per_var = y.size(1) // len(args.variables)
                        y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
                    y_te_raw = unnormalise(y_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                        norm_method=norm_method, norm_stats=norm_stats_var, logit=logit, normal=normal)
                    y_te_raw_allvars_list.append(y_te_raw)
                y_te_raw_allvars = torch.stack(y_te_raw_allvars_list, dim=1)
                
            except RuntimeError:
                pdb.set_trace()
            samples.append(gen_raw_allvars)
            trues.append(y_te_raw_allvars)
            n_batches += 1
            if n_batches > 2:
                break
    current_model.train()
    samples = torch.cat(samples, dim=0)
    trues = torch.cat(trues, dim=0)
    
    return trues, samples
            

def plot_rh(trues, samples, epoch_idx, save_dir, file_suffix = ""):
    # plot RH spatial max
    forecasts = torch.amax(samples, dim = (-3, -2))
    ground_truth = torch.amax(trues, dim = (-2, -1))
    hist, mean, variance = compute_rank_histogram(ground_truth, forecasts, axis = -1, method = "min")
    plt.bar(range(1,forecasts.shape[-1]+2), hist[0])
    plt.title("Rank histogram for spatial max")
    plt.savefig(save_dir + f"rank_hist_spatial-max_{epoch_idx}{file_suffix}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
    
    # plot RH spatial quantile
    
    #forecasts = torch.mean(samples, dim = (-3, -2))
    #ground_truth = torch.mean(trues, dim = (-2, -1))
    q = 0.99
    forecasts = torch.quantile(samples.flatten(-3, -2), q, dim = -2)
    ground_truth = torch.quantile(trues.flatten(-2, -1), q, dim = -1)
    hist, mean, variance = compute_rank_histogram(ground_truth, forecasts, axis = -1, method = "min")
    plt.bar(range(1,forecasts.shape[-1]+2), hist[0])
    plt.title("Rank histogram for spatial q0.99")
    plt.savefig(save_dir + f"rank_hist_spatial-quantile0.99_{epoch_idx}{file_suffix}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
    
    # plot RH spatial mean
    forecasts = torch.mean(samples, dim = (-3, -2))
    ground_truth = torch.mean(trues, dim = (-2, -1))
    hist, mean, variance = compute_rank_histogram(ground_truth, forecasts, axis = -1, method = "min")
    plt.bar(range(1,forecasts.shape[-1]+2), hist[0])
    plt.title("Rank histogram for spatial mean")
    plt.savefig(save_dir + f"rank_hist_spatial-mean_{epoch_idx}{file_suffix}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()

def add_one_hot(xc, one_hot_in_super=False, conv=False, x=None, one_hot_dim=0):
    # if one_hot_in_super and conv:
        # OLD & WRONG
        #one_hot = x[:, -one_hot_dim:]
        #one_hot_combined = one_hot.sum(dim=1, keepdim=True)  # Shape: (BS, 1)
        #one_hot_channel = one_hot_combined.view(xc.shape[0], 1, 1).expand(-1, 1, xc.shape[2]) # Shape: (BS, 1, image_size * image_size)
        #xc = torch.cat([xc, one_hot_channel], dim=1)  # Shape: (BS, n_channels + 1, image_size * image_size)
    if one_hot_in_super and conv:
        one_hot = x[:, -one_hot_dim:]  # Shape: (BS, one_hot_dim)
        one_hot_channels = one_hot.unsqueeze(-1).expand(-1, -1, xc.shape[2])  # Shape: (BS, one_hot_dim, image_size²)
        xc = torch.cat([xc, one_hot_channels], dim=1)  # Shape: (BS, n_channels + one_hot_dim, image_size²)

    elif one_hot_in_super and not conv:
        xc = torch.cat([xc, x[:, -one_hot_dim:]], dim=1)
    return xc

if __name__ == '__main__':

    args = get_config()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda')
    
    variables_str = '_'.join(args.variables)
    if args.kernel_size_lr == 16 and args.kernel_size_hr == 1:
        subfolder = "all"
    elif args.kernel_size_lr == 4 and args.kernel_size_hr == 1:
        subfolder = "super"
    elif args.kernel_size_lr == 16 and args.kernel_size_hr == 4:
        subfolder = "coarse"
    else:
        subfolder = f"lr{args.kernel_size_lr}_hr{args.kernel_size_hr}"
    if not args.conv and not args.conv_concat and not args.nicolai_layers:
        if args.server == "ada":
            save_dir = f"results/{args.method}/super/{subfolder}/var-{variables_str}/hidden{args.hidden_dim}_num-l-{args.num_layer}_layer-shr-{args.layer_shrinkage}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
        elif args.server == "euler":
            save_dir = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/super/{subfolder}/var-{args.variables[0]}/hidden{args.hidden_dim}_num-l-{args.num_layer}_layer-shr-{args.layer_shrinkage}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
    elif args.nicolai_layers:
        if args.server == "ada":
            #num_neighbors_ups=9,                          num_neighbours_res=25,                             map_dim=12,                             noise_dim=5,                             mlp_hidden=100,                             mlp_depth=3,                           noise_dim_mlp=0
            save_dir = f"results/{args.method}/super/{subfolder}/var-{variables_str}/loc-specific-layers_norm-out-{args.norm_method_output}{args.save_name}/"
        elif args.server == "euler":
            save_dir = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/super/{subfolder}/var-{variables_str}/loc-specific-layers_norm-out-{args.norm_method_output}_{args.save_name}/"
    elif args.conv:
        if args.server == "ada":
            save_dir = f"results/{args.method}/super/{subfolder}/var-{variables_str}/conv-{args.conv}-dim{args.conv_dim}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
        elif args.server == "euler":
            save_dir = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/super/{subfolder}/var-{variables_str}/conv-{args.conv}-dim{args.conv_dim}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
    elif args.conv_concat:
        if args.server == "ada":
            save_dir = f"results/{args.method}/super/{subfolder}/var-{variables_str}/conv-concat-{args.conv_concat}-dim{args.conv_dim}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
        elif args.server == "euler":
            save_dir = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/super/{subfolder}/var-{variables_str}/conv-concat-{args.conv_concat}-dim{args.conv_dim}_avc-c-{args.avg_constraint}_norm-out-{args.norm_method_output}{args.save_name}/"
    make_folder(save_dir)
    write_config_to_file(args, save_dir)
    
    def open_log_file(file_name):
        if args.resume_epoch > 0:
            return open(file_name, "at")
        else:
            return open(file_name, "wt")

    log_file_name = os.path.join(save_dir, 'log.txt')
    log_file = open_log_file(log_file_name)

    log_file_name = os.path.join(save_dir, 'log_raw.txt')
    log_file_raw = open_log_file(log_file_name)

    log_file_name_avc = os.path.join(save_dir, 'log_avc.txt')
    log_file_avc = open_log_file(log_file_name_avc)
    
    log_file_name_max = os.path.join(save_dir, 'log_max.txt')
    log_file_max = open_log_file(log_file_name_max)
    
    log_file_name_stats = os.path.join(save_dir, 'log_stats.txt')
    log_file_stats = open_log_file(log_file_name_stats)

    log_file_name_patch = os.path.join(save_dir, 'log_patches.txt')
    log_file_patch = open_log_file(log_file_name_patch)

    log_file_name_super = os.path.join(save_dir, 'log_super.txt')
    log_file_super = open_log_file(log_file_name_super)
        
    log_file_name_time = os.path.join(save_dir, 'log_time.txt')
    log_file_time = open_log_file(log_file_name_time)
    
    
    # get run indices also here
    if args.server == "ada":
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif args.server == "euler":
        root = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    
    #### load data
    train_loader, test_loader_in = get_data_2step_naive_avg(n_models=args.n_models, 
                                                            run_indices=args.run_indices,
                                                            variables=args.variables, 
                                                            variables_lr=args.variables_lr,
                                            batch_size=args.batch_size,
                                            norm_input=args.norm_method_input, norm_output=args.norm_method_output,
                                            sqrt_transform_in=args.sqrt_transform_in, sqrt_transform_out=args.sqrt_transform_out,
                                            kernel_size=args.kernel_size_lr, 
                                            kernel_size_hr=args.kernel_size_hr,
                                            clip_quantile=args.clip_quantile_data,
                                            tr_te_split=args.tr_te_split, 
                                            test_model_index=args.test_model_index,
                                            train_model_index=args.train_model_index,
                                            logit=args.logit_transform,
                                            normal=args.normal_transform,
                                            server=args.server,
                                            stride_lr=args.stride_lr,
                                            padding_lr=args.padding_lr,
                                            filter_outliers=args.filter_outliers,
                                            precip_zeros=args.precip_zeros)
    print('#training batches:', len(train_loader))
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:args.n_visual].to(device), xc_tr_eval[:args.n_visual].to(device), y_tr_eval[:args.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:args.n_visual].to(device), xc_te_eval[:args.n_visual].to(device), y_te_eval[:args.n_visual].to(device)
    
    #### get norm stats file
    if args.server == "euler":
        args.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    norm_stats = {}
    for i in range(len(args.variables)):
        if args.kernel_size_hr == 1:
            mode_unnorm = "hr"
        elif args.kernel_size_hr == 2:
            mode_unnorm = "hr_avg_2"
            # raise NotImplementedError("Kernel size hr: only 1 or 4 are implemented")
        elif args.kernel_size_hr == 4:
            mode_unnorm = "hr_avg_4"
        elif args.kernel_size_hr == 8:
            mode_unnorm = "hr_avg_8"
        elif args.kernel_size_hr == 16:
            mode_unnorm = "hr_avg"
        elif args.kernel_size_hr == 32:
            mode_unnorm = "hr_avg_32"
        elif args.kernel_size_hr == 64:
            mode_unnorm = "hr_avg_64"
        if args.variables[i] in ["pr", "sfcWind"] and args.sqrt_transform_out:
            name_str = "_sqrt"
        else:
            name_str = ""
        if args.kernel_size_hr == 1:
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
        else:
            norm_stats[args.variables[i]] = None
        
    #### build model
    if args.method == 'eng_2step':
        n_vars = 1
        assert args.norm_method_output != "rank_val"
        # in_dim = x_tr_eval.shape[1]
        if len(args.variables) > 1:
            out_dim = y_tr_eval.shape[-1] * len(args.variables)
        else:
            if args.fft:
                out_dim = 2 * y_tr_eval.shape[-1]
            else:
                out_dim = y_tr_eval.shape[-1]
                
        if len(args.variables) > 1:
            interm_dim = xc_tr_eval.shape[-1] * len(args.variables)
        else:
            interm_dim = xc_tr_eval.shape[-1]
        val_dim = None
        n_channels = xc_tr_eval.shape[1]
        if args.one_hot_in_super:
            one_hot_dim = 7
            interm_dim += one_hot_dim
        else:
            one_hot_dim = 0
        
        if args.add_x_in_super:
            interm_dim += 720
            assert not args.conv and not args.conv_concat
            assert len(args.variables) == 1
        
        if args.mlp_conv:
            raise NotImplementedError
        elif args.conv and args.kernel_size_lr == 16 and args.kernel_size_hr == 1:
            assert not args.conv_concat
            print("building 16x conv model")
            assert not args.one_hot_in_super # not implemented yet
            model = Generator16x(conv_dim=args.conv_dim, n_channels=n_channels).to(device)
        elif args.conv and (args.kernel_size_lr // args.kernel_size_hr == 4):
            assert not args.conv_concat
            print("building 4x conv model")
            model = Generator4x(conv_dim=args.conv_dim, n_channels=n_channels, one_hot_channel=args.one_hot_in_super, one_hot_dim=one_hot_dim,
                                image_size=128//args.kernel_size_lr).to(device)
        elif args.conv_concat and args.kernel_size_lr == 16 and args.kernel_size_hr == 1:
            raise NotImplementedError("Concatenating noise for kernel size 16 not implemented yet")
        elif args.conv_concat and (args.kernel_size_lr // args.kernel_size_hr == 4):
            assert not args.conv
            print("building 4x conv model")
            model = Generator4xConcat(conv_dim=args.conv_dim, n_channels=n_channels, one_hot_channel=args.one_hot_in_super, one_hot_dim=one_hot_dim,
                                      num_noise_channels=args.num_noise_channels,
                                      image_size=128//args.kernel_size_lr).to(device)
        elif args.conv and (args.kernel_size_lr // args.kernel_size_hr == 2):
            print("building 2x conv model")
            model = Generator2x(conv_dim=args.conv_dim, n_channels=n_channels, image_size=128//args.kernel_size_lr).to(device)
        elif args.nicolai_layers:
            if args.one_hot_in_super:
                num_classes = 8
            else:
                num_classes = 1
            model = RectUpsampleWithResiduals(128//args.kernel_size_lr, 
                            128 // args.kernel_size_hr,
                            n_features=len(args.variables),
                            num_classes=num_classes,
                            num_neighbors_ups=args.num_neighbors_ups,
                            num_neighbors_res=args.num_neighbors_res,
                            map_dim=args.latent_dim,
                            noise_dim=5,
                            mlp_hidden=args.hidden_dim,
                            mlp_depth=args.mlp_depth,
                            noise_dim_mlp=args.noise_dim_mlp,
                            double_linear=args.double_linear,
                            softmax=False,
                            split_residuals=not args.not_split_residuals
                            ).to(device)
            # change to one hot dim later
            
            if args.add_intermediate_loss:
                assert args.latent_dim == len(args.variables)
            
        else:
            print("building dense MLP model")
            model = StoUNet(interm_dim, out_dim, args.num_layer, args.hidden_dim, args.noise_dim,
                    add_bn=args.bn, out_act=args.out_act, resblock=args.mlp, noise_std=args.noise_std,
                    preproc_layer=args.preproc_layer, n_vars=n_vars,
                    layer_shrinkage=args.layer_shrinkage, dropout=args.dropout).to(device)

        model = torch.nn.DataParallel(model)
        optimizer_super = torch.optim.Adam(model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f'Built a model with #params: {count_parameters(model)}')            
    else:
        raise NotImplementedError("Method not implemented")
    
    if args.resume_epoch > 0:
        print("Resume training from epoch {}".format(args.resume_epoch))
        ckpt_dir = save_dir + f"model_{args.resume_epoch}.pt"
        model.module.load_state_dict(torch.load(ckpt_dir))
    
    
    # ----------- START TRAIN ------------------
    mp = torch.nn.MaxPool1d(128*128, stride=128*128)
    mp_patch = torch.nn.MaxPool1d(args.patch_size**2, stride=args.patch_size**2)

    # MSE
    mse = torch.nn.MSELoss(reduction='none')

    for epoch_idx in range(args.resume_epoch, args.num_epochs):
        start_time_epoch = time.time()
        if epoch_idx == args.resume_epoch:
            print('Training has started!')
        
        # if epoch_idx == args.burn_in + 1:
        #    print('Burn-in period is over, now fix parameter in super-res model!')
        #    for param in model.parameters():
        #        param.requires_grad = False
        
        loss_tr = 0; s1_tr = 0; s2_tr = 0
        loss_avc = 0
        loss_max = 0
        loss_tr_s = 0; s1_tr_s = 0; s2_tr_s = 0
        loss_tr_raw = 0; s1_tr_raw = 0; s2_tr_raw = 0
        loss_tr_locs = 0; loss_tr_batchel = 0
        loss_tr_patch = 0; loss_tr_locs_patch = 0; loss_tr_batchel_patch = 0
        n_batches = 0
        n_batches_raw = 0
        current_loader = train_loader
        for batch_idx, data_batch in enumerate(current_loader):
            optimizer_super.zero_grad()
            x, xc, y = data_batch
            x, xc, y = x.to(device), xc.to(device), y.to(device)
            x = x.view(x.shape[0], -1)

            if not args.conv and not args.conv_concat:
                y = y.view(y.shape[0], -1)
                xc = xc.view(xc.shape[0], -1)

            # pdb.set_trace()
            if args.add_x_in_super:
                xc = torch.cat([xc, x[:, :720]], dim=1)
                
            if args.nicolai_layers:
                if args.one_hot_in_super:
                    cls_ids = get_run_index_from_onehot(x[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
                    #x[:, -one_hot_dim:]
                else:
                    cls_ids = None
            if args.nicolai_layers and not args.double_linear and args.add_intermediate_loss:
                gen1, y_interm = model(xc, cls_ids=cls_ids, return_latent=True)
                gen2, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True)
            elif args.nicolai_layers and args.double_linear and args.add_intermediate_loss:
                gen1, _, y_interm = model(xc, cls_ids=cls_ids, return_latent=True)
                gen2, _, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True)
            elif args.nicolai_layers and args.add_mse_loss:
                gen1, y_upsampled = model(xc, cls_ids=cls_ids, return_mean=True)
                gen2, y_upsampled2 = model(xc, cls_ids=cls_ids, return_mean=True)
            elif args.nicolai_layers:
                gen1 = model(xc, cls_ids=cls_ids)
                gen2 = model(xc, cls_ids=cls_ids)
            else:
                gen1 = model(add_one_hot(xc, one_hot_in_super=args.one_hot_in_super, conv=(args.conv or args.conv_concat), x=x, one_hot_dim=one_hot_dim))
                gen2 = model(add_one_hot(xc, one_hot_in_super=args.one_hot_in_super, conv=(args.conv or args.conv_concat), x=x, one_hot_dim=one_hot_dim))
            
            losses = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=args.beta, patch_size=None) # loss on entire multivariate field
            
            if len(args.variables) > 1:
                losses_per_var = energy_loss_multivariate_summed(y, gen1, gen2, verbose=True, beta=args.beta, n_vars=len(args.variables))
                loss = losses[0] + losses_per_var[0]
                # ONLY FOR DEBUGGING TAKE THIS LINE: loss = losses[0]
            else:
                loss = losses[0]                
            
            if args.nicolai_layers and args.add_intermediate_loss:
                loss_interm = energy_loss_two_sample(y, y_interm, y_interm2, verbose=True, beta=args.beta, patch_size=None)
                loss += loss_interm[0] 
            elif args.nicolai_layers and args.add_mse_loss:
                loss_interm = (mse(y, y_upsampled.flatten(start_dim=1)) + mse(y, y_upsampled2.flatten(start_dim=1))) / 2
                loss_interm = loss_interm.sum(dim=1).mean()
                
                loss += args.lambda_mse_loss * loss_interm

            if torch.isnan(loss):
                print('Loss is NaN')
                pdb.set_trace()
                continue
            
            if args.avg_constraint:
                contraint = avg_constraint_per_var(xc, gen1, n_vars=len(args.variables))
                loss += contraint
            
            if args.p_norm_loss_loc:
                lossnp, lossnn = norm_loss_multivariate_summed(y, gen1, gen2, args.p_norm_loss_loc, beta_norm_loss=args.beta_norm_loss, type = "loc", agg_norm_loss="mean", n_vars = len(args.variables))
                loss_tr_locs += lossnp.item()

                if args.norm_loss_loc:
                    # old version without weighting
                    # loss += lossnp + lossnn 
                    
                    # now try weighting
                    loss += args.lambda_norm_loss_loc * (lossnp + lossnn)            
                        
                    if args.patched_loss:
                        raise NotImplementedError("Patched loss not implemented yet")
                                
            if args.p_norm_loss_batch:
                lossrn, lossrp = norm_loss_multivariate_summed(y, gen1, gen2, args.p_norm_loss_batch, beta_norm_loss=args.beta_norm_loss, type = "batch", agg_norm_loss=args.agg_norm_loss, n_vars = len(args.variables))
                loss_tr_batchel += lossrp.item()       

                if args.norm_loss_batch:
                    loss += lossrp + lossrn 
                    
                    
            loss.backward()
            optimizer_super.step()
            n_batches += 1      
            loss_tr += loss.item()
            if args.nicolai_layers and args.add_intermediate_loss and len(args.variables) > 1:
                s1_tr += (losses[1].item() + losses_per_var[1].item() + loss_interm[1].item())
                s2_tr += (losses[2].item() + losses_per_var[2].item() + loss_interm[2].item())
            elif len(args.variables) > 1:
                s1_tr += (losses[1].item() + losses_per_var[1].item())
                s2_tr += (losses[2].item() + losses_per_var[2].item())
            else:
                s1_tr += losses[1].item()
                s2_tr += losses[2].item()
            
            #loss_max += max_loss.item()
            loss_tr_s += losses[0].item()
            s1_tr_s += losses[1].item()
            s2_tr_s += losses[2].item()
            # loss_tr_patch += losses_patch[0].item()
                
            if args.kernel_size_hr == 1 and (epoch_idx == 0 or ((epoch_idx + 1) % (args.print_every_nepoch * 10) == 0)):
                if n_batches_raw < 3:    
                    n_batches_raw +=1
                    with torch.no_grad():
                        for i in range(len(args.variables)):
                            if len(y.shape) == 3:
                                y_var = y[:, i, :]
                                gen_var = gen1[:, i, :]
                                gen_var2 = gen2[:, i, :]
                            else:
                                dim_per_var = y.size(1) // len(args.variables)
                                y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen_var = gen1[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
                            
                            y_raw = unnormalise(y_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]],
                                                logit=args.logit_transform, normal=args.normal_transform)
                            gen1_raw = unnormalise(gen_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                                logit=args.logit_transform, normal=args.normal_transform)
                            gen2_raw = unnormalise(gen_var2, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                                logit=args.logit_transform, normal=args.normal_transform)
                            loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=args.beta)
                            loss_tr_raw += loss_raw.item()
                            s1_tr_raw += s1_raw.item()
                            s2_tr_raw += s2_raw.item()
        
        end_time_epoch = time.time()
        log_time = f"epoch took: {end_time_epoch - start_time_epoch}"
        log_file_time.write(log_time + '\n')
        log_file_time.flush()
           
        if (epoch_idx == 0 or (epoch_idx + 1) % args.print_every_nepoch == 0):
            log_full = f'Train [Epoch {epoch_idx + 1}] \tloss: {loss_tr / n_batches:.4f}, s1: {s1_tr / n_batches:.4f}, s2: {s2_tr / n_batches:.4f}'
            log_super = f'Train-sup [Epoch {epoch_idx + 1}] \tloss: {loss_tr_s / n_batches:.4f}, s1: {s1_tr_s / n_batches:.4f}, s2: {s2_tr_s / n_batches:.4f}'
            #log_avc = f'Train-avc [Epoch {epoch_idx + 1}] \tloss: {loss_avc / n_batches:.4f}'
            #log_max = f'Train-max [Epoch {epoch_idx + 1}] \tloss: {loss_max / n_batches:.4f}'
            log_stats = f'Train-stats [Epoch {epoch_idx + 1}] \tloss_locs: {loss_tr_locs / n_batches:.4f}, loss_batchel: {loss_tr_batchel / n_batches:.4f}'
            #log_patch = f'Train-patch [Epoch {epoch_idx + 1}] \tloss: {loss_tr_patch / n_batches:.4f}, loss_locs: {loss_tr_locs_patch / n_batches:.4f}, loss_batchel: {loss_tr_batchel_patch / n_batches:.4f}'
            if n_batches_raw > 0:
                log_raw = f'Train-raw [Epoch {epoch_idx + 1}] \tloss: {loss_tr_raw / n_batches_raw:.4f}, s1: {s1_tr_raw / n_batches_raw:.4f}, s2: {s2_tr_raw / n_batches_raw:.4f}'
                
            # ----------- GET TEST LOSS ------------------
            if epoch_idx == 0 or ((epoch_idx + 1) % (args.print_every_nepoch * 10) == 0):             
                                
                # compute test loss on normalised and original scale
                if not args.dropout:
                    model.eval()
                n_te_batches = 0
                loss_te = 0; s1_te = 0; s2_te = 0
                loss_te_s = 0; s1_te_s = 0; s2_te_s = 0
                loss_te_avc = 0
                loss_te_max = 0
                loss_te_raw = 0; s1_te_raw = 0; s2_te_raw = 0
                current_test_loader = test_loader_in
                with torch.no_grad():
                    for data_te in current_test_loader:
                        x_te, xc_te, y_te = data_te
                        x_te, xc_te, y_te = x_te.to(device), xc_te.to(device), y_te.to(device)
                        x_te = x_te.view(x_te.shape[0], -1)

                        if not args.conv and not args.conv_concat:  
                            y_te = y_te.view(y_te.shape[0], -1)
                            xc_te = xc_te.view(xc_te.shape[0], -1)
                        
                        if args.add_x_in_super:
                            xc_te = torch.cat([xc_te, x_te[:, :720]], dim=1)
                            
                        if args.nicolai_layers:
                            if args.one_hot_in_super:
                                cls_ids = get_run_index_from_onehot(x_te[:, -one_hot_dim:], gcm_dict=gcm_dict, rcm_dict=rcm_dict, rcm_list=rcm_list, gcm_list=gcm_list)
                                #x_te[:, -one_hot_dim:]
                            else:
                                cls_ids = None
                        
                        if args.nicolai_layers and not args.double_linear and args.add_intermediate_loss:
                            gen1, y_interm = model(xc_te, cls_ids=cls_ids, return_latent=True)
                            gen2, y_interm2 = model(xc_te, cls_ids=cls_ids, return_latent=True)
                        elif args.nicolai_layers and args.double_linear and args.add_intermediate_loss:
                            gen1, _, y_interm = model(xc_te, cls_ids=cls_ids, return_latent=True)
                            gen2, _, y_interm2 = model(xc_te, cls_ids=cls_ids, return_latent=True)
                        elif args.nicolai_layers:
                            gen1 = model(xc_te, cls_ids=cls_ids)
                            gen2 = model(xc_te, cls_ids=cls_ids)
                        else:
                            gen1 = model(add_one_hot(xc_te, one_hot_in_super=args.one_hot_in_super, conv=(args.conv or args.conv_concat), x=x_te, one_hot_dim=one_hot_dim))
                            gen2 = model(add_one_hot(xc_te, one_hot_in_super=args.one_hot_in_super, conv=(args.conv or args.conv_concat), x=x_te, one_hot_dim=one_hot_dim))
                        
                        if args.fft:
                            gen1 = fftpred2y(gen1)
                            gen2 = fftpred2y(gen2)
                        
                        losses = energy_loss_two_sample(y_te, gen1, gen2, verbose=True, beta=args.beta, patch_size=None)
                        
                        if len(args.variables) > 1:
                            losses_per_var = energy_loss_multivariate_summed(y_te, gen1, gen2, verbose=True, beta=args.beta, n_vars=len(args.variables))
                            loss = losses[0] + losses_per_var[0]
                        else:
                            loss = losses[0]
                            
                        if args.nicolai_layers and args.add_intermediate_loss:
                            losses_interm = energy_loss_two_sample(y_te, y_interm, y_interm2, verbose=True, beta=args.beta, patch_size=None)
                            loss += losses_interm[0]
                        
                        if args.avg_constraint:
                            constraint = avg_constraint_per_var(xc_te, gen1, n_vars=len(args.variables))
                            loss += constraint
                                        
                        #loss_te_avc += constraint.item()
                        #loss_te_max += max_loss.item()
                        loss_te += loss.item()
                        if args.nicolai_layers and args.add_intermediate_loss and len(args.variables) > 1:
                            s1_te += (losses[1].item() + losses_per_var[1].item() + losses_interm[1].item())
                            s2_te += (losses[2].item() + losses_per_var[2].item() + losses_interm[2].item())
                        elif len(args.variables) > 1:
                            s1_te += (losses[1].item() + losses_per_var[1].item())
                            s2_te += (losses[2].item() + losses_per_var[2].item())
                        else:
                            s1_te += losses[1].item()
                            s2_te += losses[2].item()
                        
                        loss_te_s += losses[0].item()
                        s1_te_s += losses[1].item()
                        s2_te_s += losses[2].item()
                        n_te_batches += 1
                        
                        if args.kernel_size_hr == 1:
                            for i in range(len(args.variables)):
                                if len(y_te.shape) == 3:
                                    y_var = y_te[:, i, :]
                                    gen_var = gen1[:, i, :]
                                    gen_var2 = gen2[:, i, :]
                                else:
                                    dim_per_var = y.size(1) // len(args.variables)
                                    y_var = y_te[:, i * dim_per_var:(i + 1) * dim_per_var]
                                    gen_var = gen1[:, i * dim_per_var:(i + 1) * dim_per_var]
                                    gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen1_raw = unnormalise(gen_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out,
                                                    norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                                    logit=args.logit_transform, normal=args.normal_transform)
                                gen2_raw = unnormalise(gen_var2, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out,
                                                    norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std,
                                                    logit=args.logit_transform, normal=args.normal_transform)
                                y_te_raw = unnormalise(y_var, mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out,
                                                    norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]],
                                                    logit=args.logit_transform, normal=args.normal_transform)
                                
                                loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_te_raw, gen1_raw, gen2_raw, verbose=True)
                                loss_te_raw += loss_raw.item()
                                s1_te_raw += s1_raw.item()
                                s2_te_raw += s2_raw.item()
                        
                        if n_te_batches > 3:
                            break
                    
                log_full += f'\nTest [Epoch {epoch_idx + 1}] \tloss: {loss_te / n_te_batches:.4f}, s1: {s1_te / n_te_batches:.4f}, s2: {s2_te / n_te_batches:.4f}'
                log_super += f'\nTest-sup [Epoch {epoch_idx + 1}] \tloss: {loss_te_s / n_te_batches:.4f}, s1: {s1_te_s / n_te_batches:.4f}, s2: {s2_te_s / n_te_batches:.4f}'
                #log_avc += f'\nTest-avc [Epoch {epoch_idx + 1}] \tloss: {loss_te_avc / n_te_batches:.4f}'
                #log_max += f'\nTest-max [Epoch {epoch_idx + 1}] \tloss: {loss_te_max / n_te_batches:.4f}'
                if n_batches_raw > 0:
                    log_raw += f'\nTest-raw [Epoch {epoch_idx + 1}] \tloss: {loss_te_raw / n_te_batches:.4f}, s1: {s1_te_raw / n_te_batches:.4f}, s2: {s2_te_raw / n_te_batches:.4f}'
                model.train()

            print(log_full)
            log_file.write(log_full + '\n')
            log_file.flush()
            
            print(log_super)
            log_file_super.write(log_super + '\n')
            log_file_super.flush()
                        
            log_file_stats.write(log_stats + '\n')
            log_file_stats.flush()            
            
            if n_batches_raw > 0:
                print(log_raw)
                log_file_raw.write(log_raw + '\n')
                log_file_raw.flush()
            
            #log_file_patch.write(log_patch + '\n')
            #log_file_patch.flush()
            
        # -------------- small little eval  -----------------------------------
        
        if (epoch_idx == 0 or (epoch_idx + 1) % args.sample_every_nepoch == 0):
            
            # loss scale
            visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_loss-scale_super', 
                          norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out,
                          logit=False, normal = False, fft=args.fft, mode_unnorm=mode_unnorm,
                          one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
            visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_loss-scale_super', 
                          norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out,
                          logit=False, normal = False, fft=args.fft, mode_unnorm=mode_unnorm,
                          one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
            
            if args.kernel_size_hr == 1: # if HR target is not coarsened also
                if args.norm_method_output == "uniform" and args.logit_transform:
                    # in this case, also visualise on uniformly transformed scale
                    visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_unif-scale_super', 
                            norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out,
                            logit = True, fft=args.fft,
                            one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
                    
                    visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_unif-scale_super', 
                            norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out,
                            logit = True, fft=args.fft,
                            one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
                
                # super model on raw data scale
                visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_super', 
                            norm_method=args.norm_method_output, norm_stats=norm_stats,
                        square_data=False, sqrt_transform=args.sqrt_transform_out,
                        logit = args.logit_transform, normal=args.normal_transform, fft=args.fft,
                        one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
                visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_super', 
                            norm_method=args.norm_method_output, norm_stats=norm_stats,
                    square_data=False, sqrt_transform=args.sqrt_transform_out,
                    logit = args.logit_transform, normal=args.normal_transform, fft=args.fft,
                    one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), x_one_hot=x_tr_eval, one_hot_in_super=args.one_hot_in_super)
            
            losses_to_img(save_dir, f"log.txt", "full", "_full")
            losses_to_img(save_dir, f"log_super.txt", "sup", "_super")
            losses_to_img(save_dir, f"log_raw.txt", "raw", "_raw")
            losses_to_img(save_dir, f"log_avc.txt", "avc", "_avc")
            
            # MULTIVARIATE 
            # only on loss scale          
            trues, samples = get_eval_samples(model.module, current_test_loader, mode_unnorm=mode_unnorm, norm_stats=None, input_mode = "xc", output_mode = "y", norm_method=None,
                                              one_hot_dim=one_hot_dim, conv=(args.conv or args.conv_concat), one_hot_in_super=args.one_hot_in_super)
            for i in range(len(args.variables)):
                plot_rh(trues[:, i, :, :], samples[:, i, :, :, :], epoch_idx, save_dir, file_suffix=f"_full-model-var-{args.variables[i]}")
            
        if epoch_idx == 0 or (epoch_idx + 1) % args.save_model_every == 0:# and i >= 30:
            torch.save(model.module.state_dict(), save_dir + f"model_{epoch_idx}.pt")
            
    # Clean up memory
    torch.cuda.empty_cache()
    