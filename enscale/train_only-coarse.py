import torch
import os
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modules import StoUNet, LinearModel, MultipleStoUNetWrapper, MeanResidualWrapper
from loss_func import energy_loss_two_sample, norm_loss_multivariate_summed

from enscale.data_old import get_data_2step_naive_avg
import argparse
from load_config import load_config
from utils import *
import sys
import pdb
sys.path.append("..")


def visual_sample(model, x, y, save_dir, cfg, norm_method=None, norm_stats=None,sqrt_transform=True, square_data=False, mode_unnorm = "hr", logit=False, xc_prev=None):
    model.eval()
    with torch.no_grad():
        if xc_prev is None:
            gen = model(x.view(x.shape[0], -1))
            gen2 = model(x.view(x.shape[0], -1))
        else:
            gen = model(torch.cat([x.view(x.shape[0], -1), xc_prev.view(xc_prev.shape[0], -1)], dim=1))
            gen2 = model(torch.cat([x.view(x.shape[0], -1), xc_prev.view(xc_prev.shape[0], -1)], dim=1))
    model.train()
    for i in range(len(cfg.data.variables)):
        if len(gen.shape) == 3:
            gen_var = gen[:, i, :]
            gen_var2 = gen2[:, i, :]
        else:
            dim_per_var = gen.size(1) // len(cfg.data.variables)
            gen_var = gen[:, i * dim_per_var:(i + 1) * dim_per_var]
            gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
            
        if len(y.shape) == 3:
            y_var = y[:, i, :]
        else:
            dim_per_var = gen.size(1) // len(cfg.data.variables)
            y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
        
        if norm_stats is not None:
            norm_stats_var = norm_stats[cfg.data.variables[i]]
        else:
            norm_stats_var = None
        gen_var = unnormalise(gen_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var, 
                    final_square=square_data, logit=logit).unsqueeze(1)
        gen_var2 = unnormalise(gen_var2, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var,
                    final_square=square_data, logit=logit).unsqueeze(1) 
        y_var = unnormalise(y_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=sqrt_transform, norm_method=norm_method, norm_stats=norm_stats_var, 
                           final_square=square_data, logit=logit).unsqueeze(1)
        
        if save_dir.endswith("_super"):
            # upsample input sample to match size of output
            s1_low = 128 / cfg.data.kernel_size_lr
            s2_low = 128 / cfg.data.kernel_size_lr
            s1 = 128
            s2 = 128
            x_ups = torch.nn.functional.interpolate(x.view(x.shape[0], int(s1_low), int(s2_low)).unsqueeze(1), size=(s1, s2), mode='nearest') # upsample again to keep size
            sample = torch.cat([x_ups.cpu(), y_var.cpu(), gen_var.cpu(), gen_var2.cpu()])
        else:
            sample = torch.cat([y_var.cpu(), gen_var.cpu(), gen_var2.cpu()])
        
        sample = torch.clamp(sample, torch.quantile(y_var, 0.0005).item(), torch.quantile(y_var, 0.9995).item())
        plt.matshow(make_grid(sample, nrow=y.shape[0]).permute(1, 2, 0)[:,:,0], cmap="rainbow"); plt.axis('off'); 
        plt.savefig(save_dir + f"_var-{cfg.data.variables[i]}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
        # save_image(sample, save_dir, normalize=True, scale_each=True)

def get_eval_samples(current_model, current_test_loader, cfg, mode_unnorm="hr", norm_method=None, norm_stats=None, input_mode = "x", output_mode = "y", logit=False, temporal=False):
    current_model.eval()
    samples = []
    trues = []
    with torch.no_grad():
        n_batches = 0
        for data_te in current_test_loader:
            if temporal:
                x_te_prev, xc_te_prev, y_te_prev, x_te, xc_te, y_te = data_te
                x_te_prev, xc_te_prev, y_te_prev, x_te, xc_te, y_te = x_te_prev.to(device), xc_te_prev.to(device), y_te_prev.to(device), x_te.to(device), xc_te.to(device), y_te.to(device)
            else:
                x_te, xc_te, y_te = data_te
                x_te, xc_te, y_te = x_te.to(device), xc_te.to(device), y_te.to(device)
            x_te = x_te.view(x_te.shape[0], -1)
            
            if temporal:
                assert input_mode == "x"
                gen = current_model.sample(torch.cat([x_te, xc_te_prev.view(xc_te_prev.shape[0], -1)], dim=1), sample_size=5)
            else:
                if input_mode == "x":
                    gen = current_model.sample(x_te, sample_size=5)
                elif input_mode == "xc":
                    gen = current_model.sample(xc_te, sample_size=5)
                
            try:                
                gen_raw_allvars_list = []
                for i in range(len(cfg.data.variables)):
                    
                    if norm_stats is not None:
                        norm_stats_var = norm_stats[cfg.data.variables[i]]
                    else:
                        norm_stats_var = None
                    
                    gen_raw_var_list = []
                    for j in range(5):
                        if len(gen.shape) == 4: # 
                            gen_var_sample = gen[:, i, :, j]
                        elif len(gen.shape) == 3:
                            dim_per_var = gen.size(1) // len(cfg.data.variables)
                            gen_var_sample = gen[:, i * dim_per_var:(i + 1) * dim_per_var, j]
                        gen_raw = unnormalise(gen_var_sample, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=cfg.preprocessing.sqrt_transform_out, 
                                            norm_method=norm_method, norm_stats=norm_stats_var, sep_mean_std=cfg.preprocessing.sep_mean_std,
                                            logit=logit)
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
                for i in range(len(cfg.data.variables)):
                    
                    if norm_stats is not None:
                        norm_stats_var = norm_stats[cfg.data.variables[i]]
                    else:
                        norm_stats_var = None
                    
                    if len(y.shape) == 3:
                        y_var = y[:, i, :]
                    elif len(y.shape) == 2:
                        dim_per_var = y.size(1) // len(cfg.data.variables)
                        y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
                    y_te_raw = unnormalise(y_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=cfg.preprocessing.sqrt_transform_out, 
                                        norm_method=norm_method, norm_stats=norm_stats_var, logit=logit)
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
    
    # plot RH spatial mean
    forecasts = torch.mean(samples, dim = (-3, -2))
    ground_truth = torch.mean(trues, dim = (-2, -1))
    hist, mean, variance = compute_rank_histogram(ground_truth, forecasts, axis = -1, method = "min")
    plt.bar(range(1,forecasts.shape[-1]+2), hist[0])
    plt.title("Rank histogram for spatial mean")
    plt.savefig(save_dir + f"rank_hist_spatial-mean_{epoch_idx}{file_suffix}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        return parser.parse_args()

    args = parse_args()
    cfg = load_config(args.config)
    
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)
    
    device = torch.device('cuda')
    
    if cfg.general.server == "euler":
        prefix = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/"
    elif cfg.general.server == "ada":
        prefix = "results/eng_2step/"
    
    variables_str = '_'.join(cfg.data.variables)
    
    if cfg.model.method == 'eng_2step':
        # save_dir = f"results/coarse/var-{cfg.data.variables[0]}{cfg.model.method}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_sqrt-{cfg.preprocessing.sqrt_transform_out}_out-act-{cfg.model.out_act}_lay-shr{cfg.model.layer_shrinkage}_models-{''.join(map(str, cfg.data.run_indices))}{cfg.general.save_name}/"
        if cfg.data.kernel_size_lr == 16:
            # save_dir = prefix + f"coarse/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_sqrt-{cfg.preprocessing.sqrt_transform_out}_out-act-{cfg.model.out_act}{cfg.general.save_name}/"
            save_dir = prefix + f"coarse/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_norm-out-{cfg.preprocessing.norm_method_output}{cfg.general.save_name}/"
        else:
            save_dir = prefix + f"coarse/kernel-{cfg.data.kernel_size_lr}/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_sqrt-{cfg.preprocessing.sqrt_transform_out}_out-act-{cfg.model.out_act}{cfg.general.save_name}/"
    elif cfg.model.method == 'nn_det' or cfg.model.method == "residual" or cfg.model.method == "residual_from_mean":
        save_dir = prefix + f"coarse/{cfg.model.method}/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_sqrt-{cfg.preprocessing.sqrt_transform_out}_out-act-{cfg.model.out_act}_lay-shr{cfg.model.layer_shrinkage}{cfg.general.save_name}/"
    elif cfg.model.method == 'linear':
        save_dir = prefix + f"coarse/var-{variables_str}/{cfg.model.method}/sqrt-{cfg.preprocessing.sqrt_transform_out}_models-{''.join(map(str, cfg.data.run_indices))}{cfg.general.save_name}/"
    elif cfg.model.method == 'eng_temporal':
        save_dir = prefix + f"coarse_temporal/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_norm-out-{cfg.preprocessing.norm_method_output}{cfg.general.save_name}/"
    make_folder(save_dir)
    write_config_to_file(cfg, save_dir)
    
    def open_log_file(file_name):
        if cfg.general.resume_epoch > 0:
            return open(file_name, "at")
        else:
            return open(file_name, "wt")

    log_file_name = os.path.join(save_dir, 'log.txt')
    log_file = open_log_file(log_file_name)

    log_file_name_coarse = os.path.join(save_dir, 'log_coarse.txt')
    log_file_coarse = open_log_file(log_file_name_coarse)

    log_file_name_mse = os.path.join(save_dir, 'log_mse.txt')
    log_file_mse = open_log_file(log_file_name_mse)
    
    log_file_name_raw = os.path.join(save_dir, 'log_raw.txt')
    log_file_raw = open_log_file(log_file_name_raw)
    
    log_file_name_stats = os.path.join(save_dir, 'log_stats.txt')
    log_file_stats = open_log_file(log_file_name_stats)
    
    # RSDS
    log_file_name_rsds = os.path.join(save_dir, 'log_rsds.txt')
    log_file_rsds = open_log_file(log_file_name_rsds)
    
    #### load data
    if cfg.model.method == "eng_temporal":
        return_timepair = True
    else:
        return_timepair = False
    train_loader, test_loader_in = get_data_2step_naive_avg(
                                            run_indices=cfg.data.run_indices,
                                            n_models=cfg.data.n_models, 
                                            variables=cfg.data.variables, variables_lr=cfg.data.variables_lr,
                                            batch_size=cfg.training.batch_size,
                                            norm_input=cfg.preprocessing.norm_method_input, norm_output=cfg.preprocessing.norm_method_output,
                                            sqrt_transform_in=cfg.preprocessing.sqrt_transform_in, sqrt_transform_out=cfg.preprocessing.sqrt_transform_out,
                                            kernel_size=cfg.data.kernel_size_lr, mask_gcm=cfg.data.mask_gcm,
                                            joint_one_hot=cfg.model.split_coarse_model,
                                            ignore_one_hot_gcm=cfg.data.ignore_one_hot_gcm,
                                            ignore_one_hot_rcm=cfg.data.ignore_one_hot_rcm,
                                            tr_te_split=cfg.data.tr_te_split, 
                                            test_size=1-cfg.data.tr_te_split_ratio,
                                            logit=cfg.preprocessing.logit_transform,
                                            normal=cfg.preprocessing.normal_transform,
                                            only_winter=cfg.data.only_winter,
                                            server=cfg.general.server,
                                            return_timepair=return_timepair,
                                            precip_zeros=cfg.data.precip_zeros)
    print('#training batches:', len(train_loader))
    
    if cfg.model.method == "eng_temporal":
        x_tr_eval_prev, xc_tr_eval_prev, y_tr_eval_prev, x_tr_eval, xc_tr_eval, y_tr_eval, = next(iter(train_loader))
        x_tr_eval_prev = x_tr_eval_prev[:cfg.general.n_visual].to(device)
        xc_tr_eval_prev = xc_tr_eval_prev[:cfg.general.n_visual].to(device)
        y_tr_eval_prev = y_tr_eval_prev[:cfg.general.n_visual].to(device)
        x_tr_eval = x_tr_eval[:cfg.general.n_visual].to(device)
        xc_tr_eval = xc_tr_eval[:cfg.general.n_visual].to(device)
        y_tr_eval = y_tr_eval[:cfg.general.n_visual].to(device)
        x_te_eval_prev, xc_te_eval_prev, y_te_eval_prev, x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
        x_te_eval_prev = x_te_eval_prev[:cfg.general.n_visual].to(device)
        xc_te_eval_prev = xc_te_eval_prev[:cfg.general.n_visual].to(device)
        y_te_eval_prev = y_te_eval_prev[:cfg.general.n_visual].to(device)
        x_te_eval = x_te_eval[:cfg.general.n_visual].to(device)
        xc_te_eval = xc_te_eval[:cfg.general.n_visual].to(device)
        y_te_eval = y_te_eval[:cfg.general.n_visual].to(device)        
        
        fig, axs = plt.subplots(3, 8, figsize=(15, 6))

        # Determine the min and max values for the color scale
        vmin = min(xc_tr_eval.min().item(), xc_tr_eval_prev.min().item())
        vmax = max(xc_tr_eval.max().item(), xc_tr_eval_prev.max().item())

        for i in range(8):
            # First row: xc_tr_eval
            axs[0, i].imshow(xc_tr_eval[i, 0, :].view(8, 8).cpu().numpy(), cmap="Spectral_r", vmin=vmin, vmax=vmax)
            axs[0, i].set_title(f'xc_tr_eval[{i}]')
            axs[0, i].axis('off')
            
            # Second row: xc_tr_eval_prev
            axs[1, i].imshow(xc_tr_eval_prev[i, 0, :].view(8, 8).cpu().numpy(), cmap="Spectral_r", vmin=vmin, vmax=vmax)
            axs[1, i].set_title(f'xc_tr_eval_prev[{i}]')
            axs[1, i].axis('off')
            
            # Third row: x_tr_eval
            axs[2, i].imshow(x_tr_eval[i, :720].view(20, 36).cpu().numpy(), cmap="Spectral_r", vmin=vmin, vmax=vmax)
            axs[2, i].set_title(f'x_tr_eval[{i}]')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir + "test_eval_samples.png", bbox_inches="tight", pad_inches=0, dpi=300)
    else:
        x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
        x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:cfg.general.n_visual].to(device), xc_tr_eval[:cfg.general.n_visual].to(device), y_tr_eval[:cfg.general.n_visual].to(device)
        x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
        x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:cfg.general.n_visual].to(device), xc_te_eval[:cfg.general.n_visual].to(device), y_te_eval[:cfg.general.n_visual].to(device)
    
    if cfg.data.kernel_size_lr == 16:
        mode_unnorm = "hr_avg"
    elif cfg.data.kernel_size_lr == 4:
        mode_unnorm = "hr_avg_4"
    elif cfg.data.kernel_size_lr == 8:
        mode_unnorm = "hr_avg_8"
    elif cfg.data.kernel_size_lr == 32:
        mode_unnorm = "hr_avg_32"
    elif cfg.data.kernel_size_lr == 64:
        mode_unnorm = "hr_avg_64"
        
    if cfg.general.server == "euler":
        cfg.data.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
        
    #### get norm stats file        
    norm_stats = {}
    for i in range(len(cfg.data.variables)):
        if cfg.data.variables[i] in ["pr", "sfcWind"] and cfg.preprocessing.sqrt_transform_out:
            name_str = "_sqrt"
        else:
            name_str = ""
        if cfg.preprocessing.norm_method_output == "normalise_pw":
            norm_stats[cfg.data.variables[i]] = None
        elif cfg.preprocessing.norm_method_output == "normalise_scalar":
            ns_path = os.path.join(cfg.data.data_dir, "norm_stats", f"hr_norm_stats_full-data_" + cfg.data.variables[i] + "_train_ALL" + name_str + ".pt")
            norm_stats[cfg.data.variables[i]] = torch.load(ns_path, map_location=device)
        # TO DO: add norm stats for uniform_per_model, or maybe update this path
        
        elif cfg.preprocessing.norm_method_output == "uniform" and mode_unnorm == "hr_avg": #"hr_norm_stats_ecdf_matrix_" + data_type + "_train_" + "ALL" + name_str + ".pt")
            name_str = ""
            ns_path = os.path.join(cfg.data.data_dir, "norm_stats", f"{mode_unnorm}8x8_norm_stats_ecdf_matrix_" + cfg.data.variables[i] + "_train_SUBSAMPLE" + name_str + ".pt")
            norm_stats[cfg.data.variables[i]] = torch.load(ns_path, map_location=device)
        else:
            norm_stats[cfg.data.variables[i]] = None
        

    #### build model
    if cfg.model.method == 'eng_2step' or cfg.model.method == 'eng_temporal':
        if cfg.data.variables_lr is not None:
            n_vars = len(cfg.data.variables_lr)
        else:
            n_vars = 5
        assert cfg.preprocessing.norm_method_output != "rank_val"
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        interm_dim = xc_tr_eval.shape[-1] * len(cfg.data.variables)
        if cfg.model.method == "eng_temporal":
            in_dim = in_dim + interm_dim
        val_dim = None
        if cfg.model.split_coarse_model:
            sto_unet_params = {
                'in_dim': in_dim-8,
                'out_dim': interm_dim,
                'num_layer': cfg.model.num_layer,
                'hidden_dim': cfg.model.hidden_dim,
                'noise_dim': cfg.model.noise_dim,
                'add_bn': cfg.model.bn,
                'out_act': cfg.model.out_act,
                'resblock': cfg.model.mlp,
                'noise_std': cfg.model.noise_std,
                'preproc_layer': False,
                'n_vars': n_vars,
                'time_dim': 5,
                'val_dim': val_dim,
                'rank_dim': 720,
                'preproc_dim': cfg.model.preproc_dim,
                'layer_shrinkage': cfg.model.layer_shrinkage
            }
            model = MultipleStoUNetWrapper(num_models=8, sto_unet_params=sto_unet_params, one_hot_dim=8).to(device)
        else:
            interm_dim_per_var = interm_dim // len(cfg.data.variables)
            if cfg.model.method == "eng_temporal":
                model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, cfg.model.noise_dim,
                                        add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                                        preproc_layer=cfg.model.preproc_layer,
                                        input_dims_for_preproc=np.array(
                                            [720  for k in range(n_vars)] +
                                            [5, 7] +
                                            [interm_dim_per_var for k in range(len(cfg.data.variables))]),
                                        preproc_dim=cfg.model.preproc_dim, layer_shrinkage=cfg.model.layer_shrinkage).to(device)
            else:
                model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, cfg.model.noise_dim,
                        add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                        preproc_layer=cfg.model.preproc_layer,
                        input_dims_for_preproc=np.array(
                            [720  for k in range(n_vars)] +
                            [5, 7]),
                        preproc_dim=cfg.model.preproc_dim, layer_shrinkage=cfg.model.layer_shrinkage).to(device)
        
        optimizer_coarse = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        print(f'Built a model with #params: {count_parameters(model)}')           

    elif cfg.model.method == 'nn_det':
        if cfg.data.variables_lr is not None:
            n_vars = len(cfg.data.variables_lr)
        else:
            n_vars = 5
        assert cfg.preprocessing.norm_method_output != "rank_val"
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        interm_dim = xc_tr_eval.shape[-1] * len(cfg.data.variables)
        val_dim = None
        # set noise dim to 0
        if cfg.model.split_coarse_model:
            sto_unet_params = {
                'in_dim': in_dim,
                'out_dim': interm_dim,
                'num_layer': cfg.model.num_layer,
                'hidden_dim': cfg.model.hidden_dim,
                'noise_dim': 0,
                'add_bn': cfg.model.bn,
                'out_act': cfg.model.out_act,
                'resblock': cfg.model.mlp,
                'noise_std': cfg.model.noise_std,
                'preproc_layer': cfg.model.preproc_layer,
                'n_vars': n_vars,
                'time_dim': 5,
                'val_dim': val_dim,
                'rank_dim': 720,
                'preproc_dim': cfg.model.preproc_dim,
                'layer_shrinkage': cfg.model.layer_shrinkage
            }            
            model = MultipleStoUNetWrapper(num_models=8, sto_unet_params=sto_unet_params).to(device)
            
            optimizer_coarse = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            print(f'Built a model with #params: {count_parameters(model)}')            
    
        else:
            if cfg.data.kernel_size_lr == 16 or cfg.data.kernel_size_lr == 32 or cfg.data.kernel_size_lr == 64:
                model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, 0,
                                 add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                                 preproc_layer=cfg.model.preproc_layer, n_vars=n_vars, time_dim=5, val_dim=val_dim, 
                                rank_dim=720, preproc_dim=cfg.model.preproc_dim, layer_shrinkage=cfg.model.layer_shrinkage).to(device)
            elif cfg.data.kernel_size_lr == 4 or cfg.data.kernel_size_lr == 8  or cfg.data.kernel_size_lr == 2:
                model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, 0,
                                add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                                preproc_layer=cfg.model.preproc_layer, n_vars=n_vars, time_dim=5, val_dim=val_dim, 
                                rank_dim=720, preproc_dim=cfg.model.preproc_dim, layer_shrinkage=cfg.model.layer_shrinkage).to(device)
        
            optimizer_coarse = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            print(f'Built a model with #params: {count_parameters(model)}')            
    
        
    elif cfg.model.method == "residual" or cfg.model.method == "residual_from_mean":
        if cfg.data.variables_lr is not None:
            n_vars = len(cfg.data.variables_lr)
        else:
            n_vars = 5
        assert cfg.preprocessing.norm_method_output != "rank_val"
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        interm_dim = xc_tr_eval.shape[-1] * len(cfg.data.variables)
        val_dim = None
        
        assert not cfg.model.split_coarse_model
        assert cfg.data.kernel_size_lr >= 16
        
        mean_model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, 0,
                                add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                                preproc_layer=cfg.model.preproc_layer, n_vars=n_vars, time_dim=5, val_dim=val_dim, 
                                rank_dim=720, preproc_dim=cfg.model.preproc_dim, layer_growth=cfg.model.layer_shrinkage).to(device)
        
        residual_model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, cfg.model.noise_dim,
                                add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                                preproc_layer=cfg.model.preproc_layer, n_vars=n_vars, time_dim=5, val_dim=val_dim, 
                                rank_dim=720, preproc_dim=cfg.model.preproc_dim, layer_growth=cfg.model.layer_shrinkage).to(device)
        
        model = MeanResidualWrapper(mean_model, residual_model).to(device)
        
        if cfg.model.method == "residual":
            optimizer_coarse = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            print(f'Built a model with #params: {count_parameters(model)}')            
    
        else:
            save_dir_nn_det = prefix + f"coarse/nn_det/var-{variables_str}/hd-{cfg.model.hidden_dim}_num-lay-{cfg.model.num_layer}_sqrt-{cfg.preprocessing.sqrt_transform_out}_out-act-{cfg.model.out_act}_lay-shr{cfg.model.layer_shrinkage}{cfg.general.save_name}/"
            ckpt_dir = save_dir_nn_det + f"model_{cfg.training.burn_in}.pt"
            model.mean_model.load_state_dict(torch.load(ckpt_dir))
            
            optimizer_coarse = torch.optim.Adam(model.residual_model.parameters(), lr=cfg.training.lr)
            print(f'Built a model with #params: {count_parameters(model)}')     
            
    elif cfg.model.method == 'linear':
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        interm_dim = xc_tr_eval.shape[-1]
        model = LinearModel(in_dim, interm_dim).to(device)

        optimizer_coarse = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        print(f'Built a model with #params: {count_parameters(model)}')            
        
    if cfg.general.resume_epoch > 0:
        print("Resume training from epoch {}".format(cfg.general.resume_epoch))
        ckpt_dir = save_dir + f"model_{cfg.general.resume_epoch}.pt"
        model.load_state_dict(torch.load(ckpt_dir))
    
    # ----------- MSE loss ---------------------
    mse = torch.nn.MSELoss()
    
    # ----------- START TRAIN ------------------
    mse = torch.nn.MSELoss()
    for epoch_idx in range(cfg.general.resume_epoch, cfg.training.num_epochs):
        if epoch_idx == cfg.general.resume_epoch:
            print('Training has started!')
        
        loss_tr = 0; s1_tr = 0; s2_tr = 0
        loss_tr_c = 0; s1_tr_c = 0; s2_tr_c = 0
        n_batches = 0
        loss_tr_raw = 0; s1_tr_raw = 0; s2_tr_raw = 0
        loss_tr_locs = 0; loss_tr_batchel = 0
        n_batches_raw = 0
        
        # debug rsds
        loss_tr_rsds = 0; s1_tr_rsds = 0; s2_tr_rsds = 0
        # pr for comparison
        loss_tr_pr = 0; s1_tr_pr = 0; s2_tr_pr = 0
        
        current_loader = train_loader
        for batch_idx, data_batch in enumerate(current_loader):
            optimizer_coarse.zero_grad()
            if cfg.model.method == "eng_temporal":
                x_prev, xc_prev, y_prev, x, xc, y = data_batch
                x_prev, xc_prev, y_prev, x, xc, y = x_prev.to(device), xc_prev.to(device), y_prev.to(device), x.to(device), xc.to(device), y.to(device)
            else:
                x, xc, y = data_batch
                x, xc, y = x.to(device), xc.to(device), y.to(device)
            y = y.view(y.shape[0], -1)
            x = x.view(x.shape[0], -1)
            xc = xc.view(xc.shape[0], -1)
            # pdb.set_trace()
            
            if cfg.model.method == "eng_2step":
                x_coarse = model(x)
                x_coarse_p = model(x)
                losses = energy_loss_two_sample(xc, x_coarse, x_coarse_p, verbose=True, beta=cfg.loss.beta)
                        
                loss = losses[0]
                s1 = losses[1]
                s2 = losses[2]
            elif cfg.model.method == "eng_temporal":
                x_coarse = model(torch.cat([x, xc_prev.view(xc_prev.shape[0], -1)], dim=1))
                x_coarse_p = model(torch.cat([x, xc_prev.view(xc_prev.shape[0], -1)], dim=1))
                losses = energy_loss_two_sample(xc, x_coarse, x_coarse_p, verbose=True, beta=cfg.loss.beta)
                        
                loss = losses[0]
                s1 = losses[1]
                s2 = losses[2]            
            elif cfg.model.method == "nn_det" or cfg.model.method == "linear":
                x_coarse = model(x)
                x_coarse_p = model(x)
                loss = mse(xc, x_coarse)
                s1 = loss
                s2 = torch.zeros_like(loss)
                losses = torch.stack([loss, s1, s2], dim = 0)
            elif cfg.model.method == "residual" or cfg.model.method == "residual_from_mean":
                mu = model.mean_model(x)
                res1 = model.residual_model(x)
                res2 = model.residual_model(x)
                x_coarse = mu + res1
                x_coarse_p = mu + res2
                
                loss = mse(xc, mu)
                losses = energy_loss_two_sample(xc - mu, res1, res2, verbose=True, beta=cfg.loss.beta) 
                if epoch_idx > cfg.training.burn_in or cfg.model.method == "residual_from_mean":
                    loss += losses[0]    
                s1 = losses[1]
                s2 = losses[2]
                
            #lossnp, lossnn, lossrp, lossrn = norm_loss(xc, x_coarse, x_coarse_p, p_norm_loss_loc=cfg.loss.p_norm_loss_loc, p_norm_loss_batch=cfg.loss.p_norm_loss_batch, 
            #                                            beta_norm_loss=cfg.loss.beta_norm_loss, agg_norm_loss=cfg.loss.agg_norm_loss)            
            lossnp, lossnn = norm_loss_multivariate_summed(xc, x_coarse, x_coarse_p, cfg.loss.p_norm_loss_loc, beta_norm_loss=cfg.loss.beta_norm_loss, type = "loc", agg_norm_loss="mean", n_vars = len(cfg.data.variables))
            lossrn, lossrp = norm_loss_multivariate_summed(xc, x_coarse, x_coarse_p, cfg.loss.p_norm_loss_batch, beta_norm_loss=cfg.loss.beta_norm_loss, type = "batch", agg_norm_loss=cfg.loss.agg_norm_loss, n_vars = len(cfg.data.variables))
            
            if cfg.loss.norm_loss_loc:
                # old version without weighting
                # loss += lossnp + lossnn 
                
                # now try weighting
                loss += cfg.loss.lambda_norm_loss_loc * (lossnp + lossnn)
                
                loss_tr_locs += lossnp.item()


            if cfg.loss.norm_loss_batch:
                loss += lossrp + lossrn
            
                loss_tr_batchel += lossrp.item()    
                
            loss.backward()
            optimizer_coarse.step()
                
            n_batches += 1
            loss_tr += loss.item()
            s1_tr += s1.item()
            s2_tr += s2.item()
            
            loss_tr_c += losses[0].item()
            s1_tr_c += losses[1].item()
            s2_tr_c += losses[2].item()
            
            # ----------- DEBUG RSDS ------------------
            
            loss_rsds, s1_rsds, s2_rsds = energy_loss_two_sample(xc[:, -64:], x_coarse[:, -64:], x_coarse_p[:, -64], verbose=True, beta=cfg.loss.beta)
            
            loss_tr_rsds += loss_rsds.item()
            s1_tr_rsds += s1_rsds.item()
            s2_tr_rsds += s2_rsds.item()
            
            # ----------- DEBUG PR ------------------
            loss_pr, s1_pr, s2_pr = energy_loss_two_sample(xc[:, 64:128], x_coarse[:, 64:128], x_coarse_p[:, 64:128], verbose=True, beta=cfg.loss.beta)
            
            loss_tr_pr += loss_pr.item()
            s1_tr_pr += s1_pr.item()
            s2_tr_pr += s2_pr.item()
            
            # ----------- GET RAW LOSS ------------------
            
            if cfg.loss.calc_raw_loss and cfg.data.kernel_size_lr == 16 and cfg.preprocessing.norm_method_output != "uniform_per_model":
                # compute loss on original scale
                # allows comparisons across different data normalisation methods
                # but requires pre-computed norm stats also on coarsened HR scale
                # also is rather slow depending on the normalisation method
                if epoch_idx == 0 or ((epoch_idx + 1) % (cfg.general.print_every_nepoch * 25) == 0):
                    if n_batches_raw < 3:    
                        n_batches_raw +=1
                        with torch.no_grad():                            
                            dim_per_var = x_coarse.size(1) // len(cfg.data.variables)
                            
                            for i in range(len(cfg.data.variables)):
                                xc_var = xc[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen1_var = x_coarse[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen2_var = x_coarse_p[:, i * dim_per_var:(i + 1) * dim_per_var]
                                
                                y_raw = unnormalise(xc_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=cfg.preprocessing.sqrt_transform_out, 
                                                    norm_method=cfg.preprocessing.norm_method_output, norm_stats=norm_stats[cfg.data.variables[i]],
                                                    logit=cfg.preprocessing.logit_transform)
                                gen1_raw = unnormalise(gen1_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=cfg.preprocessing.sqrt_transform_out, 
                                                    norm_method=cfg.preprocessing.norm_method_output, norm_stats=norm_stats[cfg.data.variables[i]], sep_mean_std=cfg.preprocessing.sep_mean_std,
                                                    logit=cfg.preprocessing.logit_transform)
                                gen2_raw = unnormalise(gen2_var, mode=mode_unnorm, data_type=cfg.data.variables[i], sqrt_transform=cfg.preprocessing.sqrt_transform_out, 
                                                    norm_method=cfg.preprocessing.norm_method_output, norm_stats=norm_stats[cfg.data.variables[i]], sep_mean_std=cfg.preprocessing.sep_mean_std,
                                                    logit=cfg.preprocessing.logit_transform)
                                
                                # for simplicity, sum across variables
                                loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=cfg.loss.beta)
                                loss_tr_raw += loss_raw.item()
                                s1_tr_raw += s1_raw.item()
                                s2_tr_raw += s2_raw.item()
            
        if (epoch_idx == 0 or (epoch_idx + 1) % cfg.general.print_every_nepoch == 0):
            log = f'Train [Epoch {epoch_idx + 1}]    \tloss: {loss_tr / n_batches:.4f}, s1: {s1_tr / n_batches:.4f}, s2: {s2_tr / n_batches:.4f}'
            log_coarse = f'Train-crs [Epoch {epoch_idx + 1}] \tloss: {loss_tr_c / n_batches:.4f}, s1: {s1_tr_c / n_batches:.4f}, s2: {s2_tr_c / n_batches:.4f}'
            log_stats = f'Train-stats [Epoch {epoch_idx + 1}] \tloss-loc: {loss_tr_locs / n_batches:.4f}, loss-batch: {loss_tr_batchel / n_batches:.4f}'
            if n_batches_raw > 0:
                log_raw = f'Train-raw [Epoch {epoch_idx + 1}] \tloss: {loss_tr_raw / n_batches_raw:.4f}, s1: {s1_tr_raw / n_batches_raw:.4f}, s2: {s2_tr_raw / n_batches_raw:.4f}'

            # RSDS
            log_rsds = f'Train-RSDS [Epoch {epoch_idx + 1}] \tloss: {loss_tr_rsds / n_batches:.4f}, s1: {s1_tr_rsds / n_batches:.4f}, s2: {s2_tr_rsds / n_batches:.4f}'
            log_pr = f'Train-PR [Epoch {epoch_idx + 1}] \tloss: {loss_tr_pr / n_batches:.4f}, s1: {s1_tr_pr / n_batches:.4f}, s2: {s2_tr_pr / n_batches:.4f}'
            
            # ----------- GET TEST LOSS ------------------
            if epoch_idx == 0 or ((epoch_idx + 1) % (cfg.general.print_every_nepoch * 5) == 0):             
                                
                # compute test loss on normalised and original scale
                model.eval()
                n_te_batches = 0
                loss_te = 0; s1_te = 0; s2_te = 0
                loss_te_c = 0; s1_te_c = 0; s2_te_c = 0
                loss_te_s = 0; s1_te_s = 0; s2_te_s = 0
                mse_te = 0
                
                # debug rsds
                loss_te_rsds = 0; s1_te_rsds = 0; s2_te_rsds = 0
                # pr for comparison
                loss_te_pr = 0; s1_te_pr = 0; s2_te_pr = 0
                
                current_test_loader = test_loader_in
                with torch.no_grad():
                    for data_te in current_test_loader:
                        if cfg.model.method == "eng_temporal":
                            x_te_prev, xc_te_prev, y_te_prev, x_te, xc_te, y_te = data_te
                            x_te_prev, xc_te_prev, y_te_prev, x_te, xc_te, y_te = x_te_prev.to(device), xc_te_prev.to(device), y_te_prev.to(device), x_te.to(device), xc_te.to(device), y_te.to(device)
                        else:    
                            x_te, xc_te, y_te = data_te
                            x_te, xc_te, y_te = x_te.to(device), xc_te.to(device), y_te.to(device)
                        y_te = y_te.view(y_te.shape[0], -1)
                        x_te = x_te.view(x_te.shape[0], -1)
                        xc_te = xc_te.view(xc_te.shape[0], -1)
                    
                        if cfg.model.method == "eng_temporal":
                            x_coarse = model(torch.cat([x_te, xc_te_prev.view(xc_te_prev.shape[0], -1)], dim=1))
                            x_coarse_p = model(torch.cat([x_te, xc_te_prev.view(xc_te_prev.shape[0], -1)], dim=1))
                        else:
                            x_coarse = model(x_te)
                            x_coarse_p = model(x_te)
                                                    
                        if cfg.model.method == "eng_2step" or cfg.model.method == "eng_temporal":
                            losses = energy_loss_two_sample(xc_te, x_coarse, x_coarse_p, verbose=True, beta=cfg.loss.beta)
                            loss = losses[0]
                            s1 = losses[1]
                            s2 = losses[2]
                        elif cfg.model.method == "nn_det":
                            loss = mse(xc_te, x_coarse)
                            s1 = loss
                            s2 = torch.zeros_like(loss)
                            losses = torch.stack([loss, s1, s2], dim = 0)
                        elif cfg.model.method == "residual" or cfg.model.method == "residual_from_mean":
                            mu = model.mean_model(x_te)
                            res1 = model.residual_model(x_te)
                            res2 = model.residual_model(x_te)

                            loss = mse(xc_te, mu)
                            losses = energy_loss_two_sample(xc_te - mu, res1, res2, verbose=True, beta=cfg.loss.beta)  
                            if epoch_idx > cfg.training.burn_in or cfg.model.method == "residual_from_mean":
                                loss += losses[0]
                            s1 = losses[1]
                            s2 = losses[2]
                            
                            # for comparison also get full energy_loss_two_sample
                            # losses = energy_loss_two_sample(xc_te, res1 + mu, res2 + mu, verbose=True, beta=cfg.loss.beta)
                            
                        # also get MSE
                        if cfg.model.method == 'eng_2step':
                            cond_mean = model.predict(x_te, sample_size=10)
                        elif cfg.model.method == "eng_temporal":
                            cond_mean = model.predict(torch.cat([x_te, xc_te_prev.view(xc_te_prev.shape[0], -1)], dim=1), sample_size=10)
                        elif cfg.model.method == 'nn_det':
                            cond_mean = model.predict(x_te, sample_size=1)
                        elif cfg.model.method == 'linear':
                            cond_mean = model(x_te)
                        elif cfg.model.method == "residual" or cfg.model.method == "residual_from_mean":
                            cond_mean = model.mean_model(x_te)
                        else:
                            raise NotImplementedError
                        mse_loss = mse(cond_mean, xc_te)
                        
                        mse_te += mse_loss.item()
                        loss_te += loss.item()
                        s1_te += s1.item()
                        s2_te += s2.item()
                        loss_te_c += losses[0].item()
                        s1_te_c += losses[1].item()
                        s2_te_c += losses[2].item()
                        
                        n_te_batches += 1
                        
                        # ----------- DEBUG RSDS ------------------
                        loss_rsds, s1_rsds, s2_rsds = energy_loss_two_sample(xc_te[:, -64:], x_coarse[:, -64:], x_coarse_p[:, -64], verbose=True, beta=cfg.loss.beta)
                        loss_te_rsds += loss_rsds.item()
                        s1_te_rsds += s1_rsds.item()
                        s2_te_rsds += s2_rsds.item()       
                        
                        # ----------- DEBUG PR ------------------
                        loss_pr, s1_pr, s2_pr = energy_loss_two_sample(xc_te[:, 64:128], x_coarse[:, 64:128], x_coarse_p[:, 64:128], verbose=True, beta=cfg.loss.beta)
                        loss_te_pr += loss_pr.item()
                        s1_te_pr += s1_pr.item()
                        s2_te_pr += s2_pr.item()      
                        
                        if n_te_batches > 3:
                            break           
                        
                log += f'\nTest [Epoch {epoch_idx + 1}]     \tloss: {loss_te / n_te_batches:.4f}, s1: {s1_te / n_te_batches:.4f}, s2: {s2_te / n_te_batches:.4f}'
                log_coarse += f'\nTest-crs [Epoch {epoch_idx + 1}] \tloss: {loss_te_c / n_te_batches:.4f}, s1: {s1_te_c / n_te_batches:.4f}, s2: {s2_te_c / n_te_batches:.4f}'
                log_mse = f'\nTest-MSE [Epoch {epoch_idx + 1}] \tloss: {mse_te / n_te_batches:.4f}'
                log_rsds += f'\nTest-RSDS [Epoch {epoch_idx + 1}] \tloss: {loss_te_rsds / n_te_batches:.4f}, s1: {s1_te_rsds / n_te_batches:.4f}, s2: {s2_te_rsds / n_te_batches:.4f}'
                log_pr += f'\nTest-PR [Epoch {epoch_idx + 1}] \tloss: {loss_te_pr / n_te_batches:.4f}, s1: {s1_te_pr / n_te_batches:.4f}, s2: {s2_te_pr / n_te_batches:.4f}'

                model.train()
            
            print(log)
            log_file.write(log + '\n')
            log_file.flush()
        
            print(log_coarse)
            log_file_coarse.write(log_coarse + '\n')
            log_file_coarse.flush()
            
            print(log_mse)
            log_file_mse.write(log_mse + '\n')
            log_file_mse.flush()
            
            log_file_stats.write(log_stats + '\n')
            log_file_stats.flush()
            
            if n_batches_raw > 0:
                print(log_raw)
                log_file_raw.write(log_raw + '\n')
                log_file_raw.flush()
                
            # RSDS
            print(log_rsds)
            log_file_rsds.write(log_rsds + '\n')
            log_file_rsds.flush()
            
            print(log_pr)
            log_file_rsds.write(log_pr + '\n')
            log_file_rsds.flush()
            
        # -------------- small little eval  -----------------------------------
        
        if (epoch_idx == 0 or (epoch_idx + 1) % cfg.general.sample_every_nepoch == 0):

            if cfg.model.method == "eng_temporal":
                xc_prev_te = xc_te_eval_prev
                xc_prev_tr = xc_tr_eval_prev
                temporal = True
            else:
                xc_prev_te = None
                xc_prev_tr = None
                temporal = False

            if cfg.model.method == "residual":
                visual_sample(model.mean_model, x_tr_eval, xc_tr_eval, cfg=cfg, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_coarse-mean_loss-scale', norm_method=None, norm_stats=None, 
                    square_data=False, sqrt_transform=cfg.preprocessing.sqrt_transform_out, mode_unnorm = mode_unnorm, logit=False)
                visual_sample(model.mean_model, x_te_eval, xc_te_eval, cfg=cfg, save_dir=save_dir + f'img_{epoch_idx + 1}_te_coarse-mean_loss-scale', norm_method=None, norm_stats=None,
                    square_data=False, sqrt_transform=cfg.preprocessing.sqrt_transform_out, mode_unnorm = mode_unnorm, logit=False)

            # coarse model, transformed scale
            visual_sample(model, x_tr_eval, xc_tr_eval, cfg=cfg, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_coarse_loss-scale', norm_method=None, norm_stats=None, 
                    square_data=False, sqrt_transform=cfg.preprocessing.sqrt_transform_out, mode_unnorm = mode_unnorm, logit=False, xc_prev=xc_prev_tr)
            visual_sample(model, x_te_eval, xc_te_eval, cfg=cfg, save_dir=save_dir + f'img_{epoch_idx + 1}_te_coarse_loss-scale', norm_method=None, norm_stats=None,
                    square_data=False, sqrt_transform=cfg.preprocessing.sqrt_transform_out, mode_unnorm = mode_unnorm, logit=False, xc_prev=xc_prev_te)

            losses_to_img(save_dir, f"log_coarse.txt", "crs", "_coarse")

            
            # MULTIVARIATE 
            # only on loss scale         
 
            trues, samples = get_eval_samples(model, current_test_loader, cfg, mode_unnorm=mode_unnorm, norm_stats=None, input_mode = "x", output_mode = "xc", norm_method=None, temporal=temporal)
            for i in range(len(cfg.data.variables)):
                plot_rh(trues[:, i, :, :], samples[:, i, :, :, :], epoch_idx, save_dir, file_suffix=f"_coarse-var-{cfg.data.variables[i]}")
                
            # ADDED TEMPORARILY
            for i in range(len(cfg.data.variables)):
                avg_daily_var = torch.var(samples[:, i, :, :, :], dim=-1).mean(dim=0)
                plt.imshow(avg_daily_var.cpu().numpy()); plt.axis('off');
                plt.savefig(save_dir + f"var-{cfg.data.variables[i]}_daily-var_{epoch_idx}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()

        if epoch_idx == 0 or (epoch_idx + 1) % cfg.training.save_model_every == 0:# and i >= 30:
            torch.save(model.state_dict(), save_dir + f"model_{epoch_idx}.pt")
            
    # Clean up memory
    torch.cuda.empty_cache()