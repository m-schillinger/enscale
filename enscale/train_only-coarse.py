from data_v2 import get_data_v2
import torch
import os
import random
import time
import matplotlib.pyplot as plt
from modules import StoUNet, LinearModel, MultipleStoUNetWrapper, MeanResidualWrapper
from loss_func import energy_loss_two_sample, norm_loss_multivariate_summed

import argparse
import load_config_v2
from utils import *
import sys
import pdb
sys.path.append("..")

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
    cfg = load_config_v2.load_config_v2(args.config)
    
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)
    
    device = torch.device('cuda')
    
    variables_str = '_'.join(cfg.data.variables)

    if cfg.model.method == 'eng_temporal':
        save_dir = build_save_dir(cfg, variables_str, stage="coarse_temporal")
    else:
        save_dir = build_save_dir(cfg, variables_str, stage="coarse")
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

    log_file_name_time = os.path.join(save_dir, 'log_time.txt')
    log_file_time = open_log_file(log_file_name_time)
    
    #### load data
    if cfg.model.method == "eng_temporal":
        return_timepair = True
    else:
        return_timepair = False
    
    train_loader, test_loader_in = get_data_v2(cfg, validation_size=0.1, shuffle=True)
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
    scheme = get_ensemble_encoding_scheme(cfg)
    gcm_dict = {}
    rcm_dict = {}
    if scheme in {"gcm", "rcm", "gcm+rcm"}:
        if getattr(cfg.data, "type", "") != "cordex_ensemble":
            raise ValueError("Ensemble encoding requires data.type='cordex_ensemble'.")
        root_for_combinations = getattr(cfg.data.cordex, "root", None) or cfg.data.data_dir
        _, _, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root_for_combinations)

    if scheme == "gcm":
        one_hot_dim = len(gcm_dict)
    elif scheme == "rcm":
        one_hot_dim = len(rcm_dict)
    elif scheme == "gcm+rcm":
        one_hot_dim = len(gcm_dict) + len(rcm_dict)
    else:
        one_hot_dim = 0
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
                                            [5, one_hot_dim] +
                                            [interm_dim_per_var for k in range(len(cfg.data.variables))]),
                                        preproc_dim=cfg.model.preproc_dim, layer_shrinkage=cfg.model.layer_shrinkage).to(device)
            else:
                model = StoUNet(in_dim, interm_dim, cfg.model.num_layer, cfg.model.hidden_dim, cfg.model.noise_dim,
                        add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                        preproc_layer=cfg.model.preproc_layer,
                        input_dims_for_preproc=np.array(
                            [720  for k in range(n_vars)] +
                            [5, one_hot_dim]),
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
            save_dir_nn_det = build_save_dir(cfg, variables_str, stage="coarse", method="nn_det")
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
        start_time_epoch = time.time()
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
            if cfg.loss.p_norm_loss_loc is not None:
                lossnp, lossnn = norm_loss_multivariate_summed(
                    xc,
                    x_coarse,
                    x_coarse_p,
                    cfg.loss.p_norm_loss_loc,
                    beta_norm_loss=cfg.loss.beta_norm_loss,
                    type="loc",
                    agg_norm_loss="mean",
                    n_vars=len(cfg.data.variables),
                )
            else:
                lossnp = torch.zeros((), device=xc.device, dtype=xc.dtype)
                lossnn = torch.zeros((), device=xc.device, dtype=xc.dtype)

            if cfg.loss.p_norm_loss_batch is not None:
                lossrn, lossrp = norm_loss_multivariate_summed(
                    xc,
                    x_coarse,
                    x_coarse_p,
                    cfg.loss.p_norm_loss_batch,
                    beta_norm_loss=cfg.loss.beta_norm_loss,
                    type="batch",
                    agg_norm_loss=cfg.loss.agg_norm_loss,
                    n_vars=len(cfg.data.variables),
                )
            else:
                lossrn = torch.zeros((), device=xc.device, dtype=xc.dtype)
                lossrp = torch.zeros((), device=xc.device, dtype=xc.dtype)
            
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
                                
                                s1, s2 = get_spatial_dims_from_mode(mode_unnorm)
                                y_raw = unnormalise(xc_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                gen1_raw = unnormalise(gen1_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                gen2_raw = unnormalise(gen2_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                
                                # for simplicity, sum across variables
                                loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=cfg.loss.beta)
                                loss_tr_raw += loss_raw.item()
                                s1_tr_raw += s1_raw.item()
                                s2_tr_raw += s2_raw.item()

                    end_time_epoch = time.time()
                    log_time = f"epoch took: {end_time_epoch - start_time_epoch}"
                    log_file_time.write(log_time + '\n')
                    log_file_time.flush()
            
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
                visual_sample(
                    model.mean_model,
                    x_tr_eval,
                    xc_tr_eval,
                    cfg=cfg,
                    save_dir=save_dir + f'img_{epoch_idx + 1}_tr_coarse-mean_loss-scale',
                    norm_stats=None,
                    mode_unnorm=mode_unnorm,
                )
                visual_sample(
                    model.mean_model,
                    x_te_eval,
                    xc_te_eval,
                    cfg=cfg,
                    save_dir=save_dir + f'img_{epoch_idx + 1}_te_coarse-mean_loss-scale',
                    norm_stats=None,
                    mode_unnorm=mode_unnorm,
                )

            # coarse model, transformed scale
            visual_sample(
                model,
                x_tr_eval,
                xc_tr_eval,
                cfg=cfg,
                save_dir=save_dir + f'img_{epoch_idx + 1}_tr_coarse_loss-scale',
                norm_stats=None,
                mode_unnorm=mode_unnorm,
                xc_prev=xc_prev_tr,
            )
            visual_sample(
                model,
                x_te_eval,
                xc_te_eval,
                cfg=cfg,
                save_dir=save_dir + f'img_{epoch_idx + 1}_te_coarse_loss-scale',
                norm_stats=None,
                mode_unnorm=mode_unnorm,
                xc_prev=xc_prev_te,
            )

            losses_to_img(save_dir, f"log_coarse.txt", "crs", "_coarse")

            
            # MULTIVARIATE 
            # only on loss scale         
 
            trues, samples = get_eval_samples(
                model,
                test_loader_in,
                cfg,
                device,
                mode_unnorm=mode_unnorm,
                norm_stats=None,
                input_mode="x",
                output_mode="xc",
                temporal=temporal,
                temporal_concat_input=temporal,
            )
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