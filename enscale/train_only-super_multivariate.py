import torch
import os
import random
from modules import StoUNet
from modules_cnn import Generator16x, Generator4x, Generator4xConcat, Generator2x
from modules_loc_variant import RectUpsampleWithResiduals
from loss_func import energy_loss_two_sample, avg_constraint_per_var, energy_loss_multivariate_summed, norm_loss_multivariate_summed

from data import get_data
import argparse
from load_config import load_config
from utils import *
import sys
import time
sys.path.append("..")

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
    
    variables_str = '_'.join(cfg.data.variables)
    if cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 1:
        subfolder = "all"
    elif cfg.data.kernel_size_lr == 4 and cfg.data.kernel_size_hr == 1:
        subfolder = "super"
    elif cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 4:
        subfolder = "coarse"
    else:
        subfolder = f"lr{cfg.data.kernel_size_lr}_hr{cfg.data.kernel_size_hr}"
    
    # Build save directory path using helper function
    save_dir = build_save_dir(cfg, variables_str, subfolder=subfolder, stage="super")
    make_folder(save_dir)
    write_config_to_file(cfg, save_dir)
    
    def open_log_file(file_name):
        if cfg.general.resume_epoch > 0:
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
    if cfg.general.server == "ada":
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif cfg.general.server == "euler":
        root = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    encoding_scheme = get_ensemble_encoding_scheme(cfg)
    
    #### load data
    train_loader, test_loader_in = get_data(cfg, test_size=0.1, shuffle=True)
    print('#training batches:', len(train_loader))
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:cfg.general.n_visual].to(device), xc_tr_eval[:cfg.general.n_visual].to(device), y_tr_eval[:cfg.general.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:cfg.general.n_visual].to(device), xc_te_eval[:cfg.general.n_visual].to(device), y_te_eval[:cfg.general.n_visual].to(device)
    
    #### get norm stats file
    if cfg.general.server == "euler":
        cfg.data.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    
    # Determine mode for unnormalisation
    mode_unnorm = get_mode_from_kernel_size(cfg.data.kernel_size_hr)
    
    # Load normalisation statistics for all variables only if full HR
    if cfg.data.kernel_size_hr == 1:
        norm_stats = load_norm_stats_for_variables(
            cfg.data.preprocessing,
            "hr",
            cfg.data.variables,
            device=device,
        )
    else:
        norm_stats = None
        
    #### build model
    if cfg.model.method == 'eng_2step':
        n_vars = 1
        assert cfg.data.preprocessing.norm_method_output != "rank_val"
        # in_dim = x_tr_eval.shape[1]
        if len(cfg.data.variables) > 1:
            out_dim = y_tr_eval.shape[-1] * len(cfg.data.variables)
        else:
            if cfg.data.preprocessing.fft:
                out_dim = 2 * y_tr_eval.shape[-1]
            else:
                out_dim = y_tr_eval.shape[-1]
                
        if len(cfg.data.variables) > 1:
            interm_dim = xc_tr_eval.shape[-1] * len(cfg.data.variables)
        else:
            interm_dim = xc_tr_eval.shape[-1]
        val_dim = None
        n_channels = xc_tr_eval.shape[1]
        if cfg.model.one_hot_in_super:
            if encoding_scheme == "gcm":
                one_hot_dim = len(gcm_dict)
            elif encoding_scheme == "rcm":
                one_hot_dim = len(rcm_dict)
            elif encoding_scheme == "gcm+rcm":
                one_hot_dim = len(gcm_dict) + len(rcm_dict)
            else:
                one_hot_dim = 0
            interm_dim += one_hot_dim
        else:
            one_hot_dim = 0
        
        if cfg.model.add_x_in_super:
            interm_dim += 720
            assert not cfg.model.conv and not cfg.model.conv_concat
            assert len(cfg.data.variables) == 1
        
        if cfg.model.conv and cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 1:
            assert not args.conv_concat
            print("building 16x conv model")
            assert not cfg.model.one_hot_in_super # not implemented yet
            model = Generator16x(conv_dim=cfg.model.conv_dim, n_channels=n_channels).to(device)
        elif cfg.model.conv and (cfg.data.kernel_size_lr // cfg.data.kernel_size_hr == 4):
            assert not args.conv_concat
            print("building 4x conv model")
            model = Generator4x(conv_dim=cfg.model.conv_dim, n_channels=n_channels, one_hot_channel=cfg.model.one_hot_in_super, one_hot_dim=one_hot_dim,
                                image_size=128//cfg.data.kernel_size_lr).to(device)
        elif cfg.model.conv_concat and cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 1:
            raise NotImplementedError("Concatenating noise for kernel size 16 not implemented yet")
        elif cfg.model.conv_concat and (cfg.data.kernel_size_lr // cfg.data.kernel_size_hr == 4):
            assert not args.conv
            print("building 4x conv model")
            model = Generator4xConcat(conv_dim=cfg.model.conv_dim, n_channels=n_channels, one_hot_channel=cfg.model.one_hot_in_super, one_hot_dim=one_hot_dim,
                                      num_noise_channels=cfg.model.num_noise_channels,
                                      image_size=128//cfg.data.kernel_size_lr).to(device)
        elif cfg.model.conv and (cfg.data.kernel_size_lr // cfg.data.kernel_size_hr == 2):
            print("building 2x conv model")
            model = Generator2x(conv_dim=cfg.model.conv_dim, n_channels=n_channels, image_size=128//cfg.data.kernel_size_lr).to(device)
        elif cfg.model.nicolai_layers:
            if cfg.model.one_hot_in_super:
                if encoding_scheme == "gcm":
                    num_classes = len(gcm_dict)
                elif encoding_scheme == "rcm":
                    num_classes = len(rcm_dict)
                elif encoding_scheme == "gcm+rcm":
                    num_classes = len(gcm_list)
                else:
                    num_classes = 1
                if cfg.model.one_hot_only_in_ups:
                    num_classes_resid = 1
                else:
                    num_classes_resid = num_classes
            else:
                num_classes = 1
                num_classes_resid = 1
            model = RectUpsampleWithResiduals(128//cfg.data.kernel_size_lr, 
                            128 // cfg.data.kernel_size_hr,
                            n_features=len(cfg.data.variables),
                            num_classes=num_classes,
                            num_classes_resid=num_classes_resid,
                            num_neighbors_ups=cfg.sparse_layers.num_neighbors_ups,
                            num_neighbors_res=cfg.sparse_layers.num_neighbors_res,
                            map_dim=cfg.sparse_layers.latent_dim,
                            noise_dim=5,
                            mlp_hidden=cfg.model.hidden_dim,
                            mlp_depth=cfg.sparse_layers.mlp_depth,
                            noise_dim_mlp=cfg.sparse_layers.noise_dim_mlp,
                            double_linear=cfg.sparse_layers.double_linear,
                            split_residuals=not cfg.sparse_layers.not_split_residuals
                            ).to(device)
            if cfg.sparse_layers.add_intermediate_loss:
                assert cfg.sparse_layers.latent_dim == len(cfg.data.variables)
            
        else:
            print("building dense MLP model")
            model = StoUNet(interm_dim, out_dim, cfg.model.num_layer, cfg.model.hidden_dim, cfg.model.noise_dim,
                    add_bn=cfg.model.bn, out_act=cfg.model.out_act, resblock=cfg.model.mlp, noise_std=cfg.model.noise_std,
                    preproc_layer=cfg.model.preproc_layer, n_vars=n_vars,
                    layer_shrinkage=cfg.model.layer_shrinkage, dropout=cfg.model.dropout).to(device)

        model = torch.nn.DataParallel(model)
        optimizer_super = torch.optim.Adam(model.module.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        print(f'Built a model with #params: {count_parameters(model)}')            
    else:
        raise NotImplementedError("Method not implemented")
    
    if cfg.general.resume_epoch > 0:
        print("Resume training from epoch {}".format(cfg.general.resume_epoch))
        ckpt_dir = save_dir + f"model_{cfg.general.resume_epoch}.pt"
        model.module.load_state_dict(torch.load(ckpt_dir))
    
    
    # ----------- START TRAIN ------------------
    mp = torch.nn.MaxPool1d(128*128, stride=128*128)
    mp_patch = torch.nn.MaxPool1d(cfg.loss.patch_size**2, stride=cfg.loss.patch_size**2)

    # MSE
    mse = torch.nn.MSELoss(reduction='none')

    for epoch_idx in range(cfg.general.resume_epoch, cfg.training.num_epochs):
        start_time_epoch = time.time()
        if epoch_idx == cfg.general.resume_epoch:
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

            if not cfg.model.conv and not cfg.model.conv_concat:
                y = y.view(y.shape[0], -1)
                xc = xc.view(xc.shape[0], -1)

            # pdb.set_trace()
            if cfg.model.add_x_in_super:
                xc = torch.cat([xc, x[:, :720]], dim=1)
                
            if cfg.model.nicolai_layers:
                if cfg.model.one_hot_in_super:
                    if one_hot_dim <= 0 or encoding_scheme is None:
                        cls_ids = None
                    elif encoding_scheme in {"gcm", "rcm"}:
                        cls_ids = torch.argmax(x[:, -one_hot_dim:], dim=1)
                    else:
                        cls_ids = torch.from_numpy(
                            get_run_index_from_onehot(
                                x[:, -one_hot_dim:],
                                gcm_dict=gcm_dict,
                                rcm_dict=rcm_dict,
                                rcm_list=rcm_list,
                                gcm_list=gcm_list,
                                mode="joint",
                            )
                        ).to(xc.device)
                    #x[:, -one_hot_dim:]
                else:
                    cls_ids = None
            if cfg.model.nicolai_layers and not cfg.sparse_layers.double_linear and cfg.sparse_layers.add_intermediate_loss:
                gen1, y_interm = model(xc, cls_ids=cls_ids, return_latent=True)
                gen2, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True)
            elif cfg.model.nicolai_layers and cfg.sparse_layers.double_linear and cfg.sparse_layers.add_intermediate_loss:
                gen1, _, y_interm = model(xc, cls_ids=cls_ids, return_latent=True)
                gen2, _, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True)
            elif cfg.model.nicolai_layers and cfg.sparse_layers.add_mse_loss:
                gen1, y_upsampled = model(xc, cls_ids=cls_ids, return_mean=True)
                gen2, y_upsampled2 = model(xc, cls_ids=cls_ids, return_mean=True)
            elif cfg.model.nicolai_layers:
                gen1 = model(xc, cls_ids=cls_ids)
                gen2 = model(xc, cls_ids=cls_ids)
            else:
                gen1 = model(add_one_hot(xc, one_hot_in_super=cfg.model.one_hot_in_super, conv=(cfg.model.conv or cfg.model.conv_concat), x=x, one_hot_dim=one_hot_dim))
                gen2 = model(add_one_hot(xc, one_hot_in_super=cfg.model.one_hot_in_super, conv=(cfg.model.conv or cfg.model.conv_concat), x=x, one_hot_dim=one_hot_dim))
            
            losses = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=cfg.loss.beta, patch_size=None) # loss on entire multivariate field
            
            if len(cfg.data.variables) > 1:
                losses_per_var = energy_loss_multivariate_summed(y, gen1, gen2, verbose=True, beta=cfg.loss.beta, n_vars=len(cfg.data.variables))
                loss = losses[0] + losses_per_var[0]
                # ONLY FOR DEBUGGING TAKE THIS LINE: loss = losses[0]
            else:
                loss = losses[0]                
            
            if cfg.model.nicolai_layers and cfg.sparse_layers.add_intermediate_loss:
                loss_interm = energy_loss_two_sample(y, y_interm, y_interm2, verbose=True, beta=cfg.loss.beta, patch_size=None)
                loss += loss_interm[0] 
            elif cfg.model.nicolai_layers and cfg.sparse_layers.add_mse_loss:
                loss_interm = (mse(y, y_upsampled.flatten(start_dim=1)) + mse(y, y_upsampled2.flatten(start_dim=1))) / 2
                loss_interm = loss_interm.sum(dim=1).mean()
                
                loss += cfg.sparse_layers.lambda_mse_loss * loss_interm
            
            if cfg.loss.avg_constraint:
                contraint = avg_constraint_per_var(xc, gen1, n_vars=len(cfg.data.variables))
                loss += contraint
            
            if cfg.loss.p_norm_loss_loc:
                lossnp, lossnn = norm_loss_multivariate_summed(y, gen1, gen2, cfg.loss.p_norm_loss_loc, beta_norm_loss=cfg.loss.beta_norm_loss, type = "loc", agg_norm_loss="mean", n_vars = len(cfg.data.variables))
                loss_tr_locs += lossnp.item()

                if cfg.loss.norm_loss_loc:
                    # old version without weighting
                    # loss += lossnp + lossnn 
                    
                    # now try weighting
                    loss += cfg.loss.lambda_norm_loss_loc * (lossnp + lossnn)            
                        
                    if cfg.loss.patched_loss:
                        raise NotImplementedError("Patched loss not implemented yet")
                                
            if cfg.loss.p_norm_loss_batch:
                lossrn, lossrp = norm_loss_multivariate_summed(y, gen1, gen2, cfg.loss.p_norm_loss_batch, beta_norm_loss=cfg.loss.beta_norm_loss, type = "batch", agg_norm_loss=cfg.loss.agg_norm_loss, n_vars = len(cfg.data.variables))
                loss_tr_batchel += lossrp.item()       

                if cfg.loss.norm_loss_batch:
                    loss += lossrp + lossrn 
                    
                    
            loss.backward()
            optimizer_super.step()
            n_batches += 1      
            loss_tr += loss.item()
            if cfg.model.nicolai_layers and cfg.sparse_layers.add_intermediate_loss and len(cfg.data.variables) > 1:
                s1_tr += (losses[1].item() + losses_per_var[1].item() + loss_interm[1].item())
                s2_tr += (losses[2].item() + losses_per_var[2].item() + loss_interm[2].item())
            elif len(cfg.data.variables) > 1:
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
                
            if cfg.data.kernel_size_hr == 1 and (epoch_idx == 0 or ((epoch_idx + 1) % (cfg.general.print_every_nepoch
 * 10) == 0)):
                if n_batches_raw < 3:    
                    n_batches_raw +=1
                    with torch.no_grad():
                        for i in range(len(cfg.data.variables)):
                            if len(y.shape) == 3:
                                y_var = y[:, i, :]
                                gen_var = gen1[:, i, :]
                                gen_var2 = gen2[:, i, :]
                            else:
                                dim_per_var = y.size(1) // len(cfg.data.variables)
                                y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen_var = gen1[:, i * dim_per_var:(i + 1) * dim_per_var]
                                gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
                            
                            s1, s2 = get_spatial_dims_from_mode(mode_unnorm)
                            y_raw = unnormalise(y_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                            gen1_raw = unnormalise(gen_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                            gen2_raw = unnormalise(gen_var2, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                            loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=cfg.loss.beta)
                            loss_tr_raw += loss_raw.item()
                            s1_tr_raw += s1_raw.item()
                            s2_tr_raw += s2_raw.item()
        
        end_time_epoch = time.time()
        log_time = f"epoch took: {end_time_epoch - start_time_epoch}"
        log_file_time.write(log_time + '\n')
        log_file_time.flush()
           
        if (epoch_idx == 0 or (epoch_idx + 1) % cfg.general.print_every_nepoch == 0):
            log_full = f'Train [Epoch {epoch_idx + 1}] \tloss: {loss_tr / n_batches:.4f}, s1: {s1_tr / n_batches:.4f}, s2: {s2_tr / n_batches:.4f}'
            log_super = f'Train-sup [Epoch {epoch_idx + 1}] \tloss: {loss_tr_s / n_batches:.4f}, s1: {s1_tr_s / n_batches:.4f}, s2: {s2_tr_s / n_batches:.4f}'
            #log_avc = f'Train-avc [Epoch {epoch_idx + 1}] \tloss: {loss_avc / n_batches:.4f}'
            #log_max = f'Train-max [Epoch {epoch_idx + 1}] \tloss: {loss_max / n_batches:.4f}'
            log_stats = f'Train-stats [Epoch {epoch_idx + 1}] \tloss_locs: {loss_tr_locs / n_batches:.4f}, loss_batchel: {loss_tr_batchel / n_batches:.4f}'
            #log_patch = f'Train-patch [Epoch {epoch_idx + 1}] \tloss: {loss_tr_patch / n_batches:.4f}, loss_locs: {loss_tr_locs_patch / n_batches:.4f}, loss_batchel: {loss_tr_batchel_patch / n_batches:.4f}'
            if n_batches_raw > 0:
                log_raw = f'Train-raw [Epoch {epoch_idx + 1}] \tloss: {loss_tr_raw / n_batches_raw:.4f}, s1: {s1_tr_raw / n_batches_raw:.4f}, s2: {s2_tr_raw / n_batches_raw:.4f}'
                
            # ----------- GET TEST LOSS ------------------
            if epoch_idx == 0 or ((epoch_idx + 1) % (cfg.general.print_every_nepoch * 10) == 0):             
                                
                # compute test loss on normalised and original scale
                if not cfg.model.dropout:
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

                        if not cfg.model.conv and not cfg.model.conv_concat:  
                            y_te = y_te.view(y_te.shape[0], -1)
                            xc_te = xc_te.view(xc_te.shape[0], -1)
                        
                        if cfg.model.add_x_in_super:
                            xc_te = torch.cat([xc_te, x_te[:, :720]], dim=1)
                            
                        if cfg.model.nicolai_layers:
                            if cfg.model.one_hot_in_super:
                                if one_hot_dim <= 0 or encoding_scheme is None:
                                    cls_ids = None
                                elif encoding_scheme in {"gcm", "rcm"}:
                                    cls_ids = torch.argmax(x_te[:, -one_hot_dim:], dim=1)
                                else:
                                    cls_ids = torch.from_numpy(
                                        get_run_index_from_onehot(
                                            x_te[:, -one_hot_dim:],
                                            gcm_dict=gcm_dict,
                                            rcm_dict=rcm_dict,
                                            rcm_list=rcm_list,
                                            gcm_list=gcm_list,
                                            mode="joint",
                                        )
                                    ).to(xc_te.device)
                            else:
                                cls_ids = None
                        
                        if cfg.model.nicolai_layers and not cfg.sparse_layers.double_linear and cfg.sparse_layers.add_intermediate_loss:
                            gen1, y_interm = model(xc_te, cls_ids=cls_ids, return_latent=True)
                            gen2, y_interm2 = model(xc_te, cls_ids=cls_ids, return_latent=True)
                        elif cfg.model.nicolai_layers and cfg.sparse_layers.double_linear and cfg.sparse_layers.add_intermediate_loss:
                            gen1, _, y_interm = model(xc_te, cls_ids=cls_ids, return_latent=True)
                            gen2, _, y_interm2 = model(xc_te, cls_ids=cls_ids, return_latent=True)
                        elif cfg.model.nicolai_layers:
                            gen1 = model(xc_te, cls_ids=cls_ids)
                            gen2 = model(xc_te, cls_ids=cls_ids)
                        else:
                            gen1 = model(add_one_hot(xc_te, one_hot_in_super=cfg.model.one_hot_in_super, conv=(cfg.model.conv or cfg.model.conv_concat), x=x_te, one_hot_dim=one_hot_dim))
                            gen2 = model(add_one_hot(xc_te, one_hot_in_super=cfg.model.one_hot_in_super, conv=(cfg.model.conv or cfg.model.conv_concat), x=x_te, one_hot_dim=one_hot_dim))
                        
                        if cfg.data.preprocessing.fft:
                            gen1 = fftpred2y(gen1)
                            gen2 = fftpred2y(gen2)
                        
                        losses = energy_loss_two_sample(y_te, gen1, gen2, verbose=True, beta=cfg.loss.beta, patch_size=None)
                        
                        if len(cfg.data.variables) > 1:
                            losses_per_var = energy_loss_multivariate_summed(y_te, gen1, gen2, verbose=True, beta=cfg.loss.beta, n_vars=len(cfg.data.variables))
                            loss = losses[0] + losses_per_var[0]
                        else:
                            loss = losses[0]
                            
                        if cfg.model.nicolai_layers and cfg.sparse_layers.add_intermediate_loss:
                            losses_interm = energy_loss_two_sample(y_te, y_interm, y_interm2, verbose=True, beta=cfg.loss.beta, patch_size=None)
                            loss += losses_interm[0]
                        
                        if cfg.loss.avg_constraint:
                            constraint = avg_constraint_per_var(xc_te, gen1, n_vars=len(cfg.data.variables))
                            loss += constraint
                                        
                        #loss_te_avc += constraint.item()
                        #loss_te_max += max_loss.item()
                        loss_te += loss.item()
                        if cfg.model.nicolai_layers and cfg.sparse_layers.add_intermediate_loss and len(cfg.data.variables) > 1:
                            s1_te += (losses[1].item() + losses_per_var[1].item() + losses_interm[1].item())
                            s2_te += (losses[2].item() + losses_per_var[2].item() + losses_interm[2].item())
                        elif len(cfg.data.variables) > 1:
                            s1_te += (losses[1].item() + losses_per_var[1].item())
                            s2_te += (losses[2].item() + losses_per_var[2].item())
                        else:
                            s1_te += losses[1].item()
                            s2_te += losses[2].item()
                        
                        loss_te_s += losses[0].item()
                        s1_te_s += losses[1].item()
                        s2_te_s += losses[2].item()
                        n_te_batches += 1
                        
                        if cfg.data.kernel_size_hr == 1:
                            for i in range(len(cfg.data.variables)):
                                if len(y_te.shape) == 3:
                                    y_var = y_te[:, i, :]
                                    gen_var = gen1[:, i, :]
                                    gen_var2 = gen2[:, i, :]
                                else:
                                    dim_per_var = y.size(1) // len(cfg.data.variables)
                                    y_var = y_te[:, i * dim_per_var:(i + 1) * dim_per_var]
                                    gen_var = gen1[:, i * dim_per_var:(i + 1) * dim_per_var]
                                    gen_var2 = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]
                                
                                s1, s2 = get_spatial_dims_from_mode(mode_unnorm)
                                gen1_raw = unnormalise(gen_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                gen2_raw = unnormalise(gen_var2, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                y_te_raw = unnormalise(y_var, var=cfg.data.variables[i], mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[cfg.data.variables[i]])
                                
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
        
        if (epoch_idx == 0 or (epoch_idx + 1) % cfg.general.sample_every_nepoch == 0):
            
            # loss scale
            visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_loss-scale_super',
                          cfg=cfg, norm_stats=None, mode_unnorm=mode_unnorm,
                          one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_tr_eval,
                          one_hot_in_super=cfg.model.one_hot_in_super,
                          gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
            visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_loss-scale_super',
                          cfg=cfg, norm_stats=None, mode_unnorm=mode_unnorm,
                          one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_te_eval,
                          one_hot_in_super=cfg.model.one_hot_in_super,
                          gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
            
            if cfg.data.kernel_size_hr == 1: # if HR target is not coarsened also
                if cfg.data.preprocessing.norm_method_output == "uniform" and cfg.data.preprocessing.logit_transform:
                    # in this case, also visualise on uniformly transformed scale
                        visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_unif-scale_super',
                            cfg=cfg, norm_stats=None, mode_unnorm=mode_unnorm,
                            one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_tr_eval,
                            one_hot_in_super=cfg.model.one_hot_in_super,
                            gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
                    
                        visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_unif-scale_super',
                            cfg=cfg, norm_stats=None, mode_unnorm=mode_unnorm,
                            one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_te_eval,
                            one_hot_in_super=cfg.model.one_hot_in_super,
                            gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
                
                # super model on raw data scale
                visual_sample(model, xc_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_super',
                            cfg=cfg, norm_stats=norm_stats, mode_unnorm=mode_unnorm,
                            one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_tr_eval,
                            one_hot_in_super=cfg.model.one_hot_in_super,
                            gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
                visual_sample(model, xc_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_super',
                        cfg=cfg, norm_stats=norm_stats, mode_unnorm=mode_unnorm,
                        one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat), x_one_hot=x_te_eval,
                        one_hot_in_super=cfg.model.one_hot_in_super,
                        gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
            
            losses_to_img(save_dir, f"log.txt", "full", "_full")
            losses_to_img(save_dir, f"log_super.txt", "sup", "_super")
            losses_to_img(save_dir, f"log_raw.txt", "raw", "_raw")
            losses_to_img(save_dir, f"log_avc.txt", "avc", "_avc")
            
            # MULTIVARIATE 
            # only on loss scale          
            trues, samples = get_eval_samples(model.module, test_loader_in, cfg, device,
                                              mode_unnorm=mode_unnorm, norm_stats=None, input_mode="xc", output_mode="y",
                                              one_hot_dim=one_hot_dim, conv=(cfg.model.conv or cfg.model.conv_concat),
                                              one_hot_in_super=cfg.model.one_hot_in_super,
                                              gcm_dict=gcm_dict, rcm_dict=rcm_dict, gcm_list=gcm_list, rcm_list=rcm_list)
            for i in range(len(cfg.data.variables)):
                plot_rh(trues[:, i, :, :], samples[:, i, :, :, :], epoch_idx, save_dir, file_suffix=f"_full-model-var-{cfg.data.variables[i]}")
            
        if epoch_idx == 0 or (epoch_idx + 1) % cfg.training.save_model_every == 0:# and i >= 30:
            torch.save(model.module.state_dict(), save_dir + f"model_{epoch_idx}.pt")
            
    # Clean up memory
    torch.cuda.empty_cache()
    