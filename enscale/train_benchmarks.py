import torch
import os
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modules import StoUNet, RankValModel, LinearModel, MultivariateStoUNetWrapper, StoUNetNoiseEnd
from loss_func import energy_loss_two_sample, energy_loss_rk_val_wrapper, ridge_loss, crps_pixelwise

from data import get_data, get_data_2step_naive_avg
from config import get_config
from utils import *
import sys
import pdb
sys.path.append("..")

    
def visual_sample(model, x, y, save_dir, norm_method=None, norm_stats=None, sqrt_transform=True, square_data=False, mode_unnorm = "hr", 
                  logit=False):
    model.eval()
    with torch.no_grad():
        gen1 = model(x.view(x.shape[0], -1))
        gen2 = model(x.view(x.shape[0], -1)) 
    
    y = y.view(y.shape[0], -1)
    for i, var in enumerate(args.variables): 
        if norm_stats is not None:
            norm_stats_var = norm_stats[args.variables[i]]
        else:
            norm_stats_var = None
        y_var = unnormalise(y[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, 
                            norm_method=norm_method, norm_stats=norm_stats_var, logit=logit, final_square=square_data).unsqueeze(1)
        gen1_var = unnormalise(gen1[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, 
                            norm_method=norm_method, norm_stats=norm_stats_var, sep_mean_std=args.sep_mean_std, logit=logit, final_square=square_data).unsqueeze(1)
        gen2_var = unnormalise(gen2[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=sqrt_transform, 
                            norm_method=norm_method, norm_stats=norm_stats_var, sep_mean_std=args.sep_mean_std, logit=logit, final_square=square_data).unsqueeze(1)

        sample = torch.cat([y_var.cpu(), gen1_var.cpu(), gen2_var.cpu()])
        
        if norm_method is not None:
            sample = torch.clamp(sample, torch.quantile(y_var, 0.001).item(), torch.quantile(y_var, 0.999).item())
        plt.matshow(make_grid(sample, nrow=y.shape[0]).permute(1, 2, 0)[:,:,0], cmap="rainbow"); plt.axis('off'); 
        plt.savefig(save_dir + f"_{var}.png", bbox_inches="tight", pad_inches=0, dpi=300); plt.close()
        # save_image(sample, save_dir, normalize=True, scale_each=True)
    model.train()

if __name__ == '__main__':

    args = get_config()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda')
    
    if args.server == "euler":
        prefix = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/"
    elif args.server == "ada":
        prefix = "results/"
    
    variables_str = '_'.join(args.variables)
    save_dir = prefix + f"{args.method}/var-{variables_str}/hidden{args.hidden_dim}_norm-in-{args.norm_method_input}_norm-out-{args.norm_method_output}{args.save_name}/"
    make_folder(save_dir)
    write_config_to_file(args, save_dir)
    
    log_file_name = os.path.join(save_dir, 'log.txt')
    if args.resume_epoch > 0:
        log_file = open(log_file_name, "at")
    else:
        log_file = open(log_file_name, "wt")
    
    log_file_name_raw = os.path.join(save_dir, 'log_raw.txt')
    if args.resume_epoch > 0:
        log_file_raw = open(log_file_name_raw, "at")
    else:
        log_file_raw = open(log_file_name_raw, "wt")
    
    if args.norm_method_output == "rank_val":
        log_file_name_rk = os.path.join(save_dir, 'log_rk.txt')
        if args.resume_epoch > 0:
            log_file_rk = open(log_file_name_rk, "at")
        else:
            log_file_rk = open(log_file_name_rk, "wt")
        log_file_name_val = os.path.join(save_dir, 'log_val.txt')
        if args.resume_epoch > 0:
            log_file_val = open(log_file_name_val, "at")
        else:
            log_file_val = open(log_file_name_val, "wt")
        
    
    #### load data
    
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
    print('#training batches:', len(train_loader))
    
    x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
    x_tr_eval, xc_tr_eval, y_tr_eval = x_tr_eval[:args.n_visual].to(device), xc_tr_eval[:args.n_visual].to(device), y_tr_eval[:args.n_visual].to(device)
    x_te_eval, xc_te_eval, y_te_eval = next(iter(test_loader_in))
    x_te_eval, xc_te_eval, y_te_eval = x_te_eval[:args.n_visual].to(device), xc_te_eval[:args.n_visual].to(device), y_te_eval[:args.n_visual].to(device)
    
    if args.server == "euler":
        args.data_dir = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    
    #### get norm stats file
    norm_stats = {}
    for i in range(len(args.variables)):
        if args.predict_lr:
            mode_unnorm = "lr"
        else:
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
        
    #### build model
    if args.method == 'eng_unet' or args.method == "nn_det" or args.method == "crps_pw":
        if args.variables_lr is not None:
            n_vars = len(args.variables_lr)
        else:
            if args.coarsened_hr:
                n_vars = 2
            else:
                n_vars = 5
        if args.norm_method_output != "rank_val":
            in_dim = x_tr_eval.shape[1]
            out_dim = y_tr_eval.shape[-2] * y_tr_eval.shape[-1]
            if args.norm_method_input == "rank_val":
                val_dim = (in_dim - 6 - 7- n_vars*20*36) // n_vars # subtract time-dim, one-hot-dim and n_vars*rank_dim, then express per var
            else:
                val_dim = None
            model = StoUNet(in_dim, out_dim, args.num_layer, args.hidden_dim, args.noise_dim,
            # StoUNetNoiseEnd
            # model = StoUNetNoiseEnd(in_dim, out_dim, args.num_layer, args.hidden_dim, args.noise_dim,
                            add_bn=args.bn, out_act=args.out_act, resblock=args.mlp, noise_std=args.noise_std,
                            preproc_layer=args.preproc_layer, n_vars=n_vars, time_dim=6, val_dim=val_dim, rank_dim=720, preproc_dim=20).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f'Built a model with #params: {count_parameters(model)}')
    
    elif args.method == "nn_det_per_variable" or args.method == "crps_pw_per_variable" or args.method == "eng_unet_per_variable":
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
            'preproc_layer': False,
            'n_vars': 1,
            'layer_shrinkage': args.layer_shrinkage,
        }
        model = MultivariateStoUNetWrapper(len(args.variables), super_model_params, expand_variables=False, split_input=False).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f'Built a model with #params: {count_parameters(model)}')
    
    if args.method == 'linear':
        in_dim = x_tr_eval.shape[1]
        out_dim = y_tr_eval.shape[-1]
        model = LinearModel(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f'Built a model with #params: {count_parameters(model)}')
    
    if args.resume_epoch > 0:
        print("Resume training from epoch {}".format(args.resume_epoch))
        ckpt_dir = save_dir + f"model_{args.resume_epoch}.pt"
        model.load_state_dict(torch.load(ckpt_dir))
    
    model = torch.nn.DataParallel(model)
    
    # ----------- START TRAIN ------------------
    mse = torch.nn.MSELoss()
    for epoch_idx in range(args.resume_epoch, args.num_epochs):
        if epoch_idx == args.resume_epoch:
            print('Training has started!')
        
        loss_tr = 0; s1_tr = 0; s2_tr = 0
        loss_tr_raw = 0; s1_tr_raw = 0; s2_tr_raw = 0
        if args.norm_method_output == "rank_val":
            loss_tr_rk = 0; s1_tr_rk = 0; s2_tr_rk = 0
            loss_tr_val = 0; s1_tr_val = 0; s2_tr_val = 0
        n_batches = 0
        n_batches_raw = 0
        current_loader = train_loader
        for batch_idx, data_batch in enumerate(current_loader):
            optimizer.zero_grad()
            x, xc, y = data_batch
            x, y = x.to(device), y.to(device)
            # pdb.set_trace()
            gen1 = model(x)
            gen2 = model(x)
            y = y.view(y.shape[0], -1)
            if args.method == 'linear':
                loss = ridge_loss(y, gen1, torch.nn.MSELoss(), model, alpha=args.alpha)
                s1 = loss
                s2 = torch.zeros_like(loss)
            elif args.method == "nn_det" or args.method == "nn_det_per_variable":
                loss = mse(y, gen1)
                s1 = loss
                s2 = torch.zeros_like(loss)
            elif args.method == "crps_pw" or args.method == "crps_pw_per_variable":
                loss, s1, s2 = crps_pixelwise(y, gen1, gen2, beta=args.beta, verbose=True)
            elif args.method == "eng_unet" or args.method == "eng_unet_per_variable":
                if args.norm_method_output != "rank_val":
                    loss, s1, s2 = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=args.beta)
                else:
                    losses = energy_loss_rk_val_wrapper(y, gen1, gen2, beta=args.beta, verbose=True, 
                                                        log_odds_transform = args.log_odds_transform, sep_mean_std = args.sep_mean_std)
                    loss = losses[0] + losses[3]
                    s1 = losses[1] + losses[4]
                    s2 = losses[2] + losses[5]
            loss.backward()
            optimizer.step()
            loss_tr += loss.item()
            s1_tr += s1.item()
            s2_tr += s2.item()
            if args.norm_method_output == "rank_val":
                loss_tr_rk += losses[0].item()
                s1_tr_rk += losses[1].item()
                s2_tr_rk += losses[2].item()
                loss_tr_val += losses[3].item()
                s1_tr_val += losses[4].item()
                s2_tr_val += losses[5].item()
            n_batches += 1
            with torch.no_grad():
                if n_batches_raw < 3:    
                    n_batches_raw +=1
                    for i, variables in enumerate(args.variables):
                        y_raw = unnormalise(y[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], logit=args.logit_transform)
                        gen1_raw = unnormalise(gen1[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std, logit=args.logit_transform)
                        gen2_raw = unnormalise(gen2[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                            norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std, logit=args.logit_transform)
                        loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=args.beta)
                        loss_tr_raw += loss_raw.item()
                        s1_tr_raw += s1_raw.item()
                        s2_tr_raw += s2_raw.item()
            
        if (epoch_idx == 0 or (epoch_idx + 1) % args.print_every_nepoch == 0):
            log = f'Train [Epoch {epoch_idx + 1}]    \tloss: {loss_tr / n_batches:.4f}, s1: {s1_tr / n_batches:.4f}, s2: {s2_tr / n_batches:.4f}'
            log_raw = f'Train-raw [Epoch {epoch_idx + 1}]\tloss: {loss_tr_raw / n_batches_raw:.4f}, s1: {s1_tr_raw / n_batches_raw:.4f}, s2: {s2_tr_raw / n_batches_raw:.4f}'
            
            if args.norm_method_output == "rank_val":
                log_rk = f'Train-rank [Epoch {epoch_idx + 1}] \tloss: {loss_tr_rk / n_batches:.4f}, s1: {s1_tr_rk / n_batches:.4f}, s2: {s2_tr_rk / n_batches:.4f}'
                log_val = f'Train-val [Epoch {epoch_idx + 1}] \tloss: {loss_tr_val / n_batches:.4f}, s1: {s1_tr_val / n_batches:.4f}, s2: {s2_tr_val / n_batches:.4f}'
            
            # ----------- GET TEST LOSS ------------------
            if epoch_idx == 0 or ((epoch_idx + 1) % (args.print_every_nepoch * 5) == 0):             
                
                # compute test loss on normalised and original scale
                model.eval()
                n_te_batches = 0
                loss_te = 0; s1_te = 0; s2_te = 0
                loss_te_raw = 0; s1_te_raw = 0; s2_te_raw = 0
                if args.norm_method_output == "rank_val":
                    loss_te_rk = 0; s1_te_rk = 0; s2_te_rk = 0
                    loss_te_val = 0; s1_te_val = 0; s2_te_val = 0
                current_test_loader = test_loader_in
                with torch.no_grad():
                    for data_te in current_test_loader:
                        x_te, xc_te, y_te = data_te
                        x_te, y_te = x_te.to(device), y_te.to(device)
                        gen1 = model(x_te.view(x_te.shape[0], -1))
                        gen2 = model(x_te.view(x_te.shape[0], -1))
                        y_te = y_te.view(y_te.shape[0], -1)
                        if args.method == 'linear':
                            loss = ridge_loss(y_te, gen1, torch.nn.MSELoss(), model, alpha=args.alpha)
                            s1 = loss
                            s2 = torch.zeros_like(loss)
                        elif args.method == "nn_det" or args.method == "nn_det_per_variable":
                            loss = mse(y_te, gen1)
                            s1 = loss
                            s2 = torch.zeros_like(loss)
                        elif args.method == "crps_pw" or args.method == "crps_pw_per_variable":
                            loss, s1, s2 = crps_pixelwise(y_te, gen1, gen2, beta=args.beta, verbose=True)
                        elif args.method == "eng_unet" or args.method == "eng_unet_per_variable":
                            if args.norm_method_output != "rank_val":
                                loss, s1, s2 = energy_loss_two_sample(y_te, gen1, gen2, verbose=True)
                            else:
                                losses = energy_loss_rk_val_wrapper(y_te, gen1, gen2, beta=args.beta, verbose=True,
                                                                    log_odds_transform = args.log_odds_transform, sep_mean_std = args.sep_mean_std)
                                loss = losses[0] + losses[3]
                                s1 = losses[1] + losses[4]
                                s2 = losses[2] + losses[5]
                        loss_te += loss.item()
                        s1_te += s1.item()
                        s2_te += s2.item()
                        if args.norm_method_output == "rank_val":
                            loss_te_rk += losses[0].item()
                            s1_te_rk += losses[1].item()
                            s2_te_rk += losses[2].item()
                            loss_te_val += losses[3].item()
                            s1_te_val += losses[4].item()
                            s2_te_val += losses[5].item()
                        for i, variables in enumerate(args.variables):
                            y_te_raw = unnormalise(y_te[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], logit=args.logit_transform)
                            gen1_raw = unnormalise(gen1[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std, logit=args.logit_transform)
                            gen2_raw = unnormalise(gen2[:, i*128*128:(i+1)*128*128], mode=mode_unnorm, data_type=args.variables[i], sqrt_transform=args.sqrt_transform_out, 
                                                norm_method=args.norm_method_output, norm_stats=norm_stats[args.variables[i]], sep_mean_std=args.sep_mean_std, logit=args.logit_transform)
                            loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_te_raw, gen1_raw, gen2_raw, verbose=True)
                            loss_te_raw += loss_raw.item()
                            s1_te_raw += s1_raw.item()
                            s2_te_raw += s2_raw.item()
                        n_te_batches += 1
                        if n_te_batches == 3:
                            break
                        
                log += f'\nTest [Epoch {epoch_idx + 1}]     \tloss: {loss_te / n_te_batches:.4f}, s1: {s1_te / n_te_batches:.4f}, s2: {s2_te / n_te_batches:.4f}'
                log_raw += f'\nTest-raw [Epoch {epoch_idx + 1}] \tloss: {loss_te_raw / n_te_batches:.4f}, s1: {s1_te_raw / n_te_batches:.4f}, s2: {s2_te_raw / n_te_batches:.4f}'
                if args.norm_method_output == "rank_val":
                    log_rk += f'\nTest-rank [Epoch {epoch_idx + 1}] \tloss: {loss_te_rk / n_te_batches:.4f}, s1: {s1_te_rk / n_te_batches:.4f}, s2: {s2_te_rk / n_te_batches:.4f}'
                    log_val += f'\nTest-val [Epoch {epoch_idx + 1}] \tloss: {loss_te_val / n_te_batches:.4f}, s1: {s1_te_val / n_te_batches:.4f}, s2: {s2_te_val / n_te_batches:.4f}'
                
                model.train()
            
            print(log)
            log_file.write(log + '\n')
            log_file.flush()
            
            print(log_raw)
            log_file_raw.write(log_raw + '\n')
            log_file_raw.flush()
        
            if args.norm_method_output == "rank_val":
                print(log_rk)
                log_file_rk.write(log_rk + '\n')
                log_file_rk.flush()
                print(log_val)
                log_file_val.write(log_val + '\n')
                log_file_val.flush()
            
        # -------------- small little eval  -----------------------------------
        
        if (epoch_idx == 0 or (epoch_idx + 1) % args.sample_every_nepoch == 0):
            
            visual_sample(model, x_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr', norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out)
            visual_sample(model, x_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te', norm_method=None, norm_stats=None, square_data=False, sqrt_transform=args.sqrt_transform_out)
        
            
            # visual eval on original scale
            visual_sample(model, x_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_raw', norm_method=args.norm_method_output, norm_stats=norm_stats,  
                         square_data=True, sqrt_transform=args.sqrt_transform_out, logit=args.logit_transform)
            visual_sample(model, x_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_raw', norm_method=args.norm_method_output, norm_stats=norm_stats, 
                          square_data=True, sqrt_transform=args.sqrt_transform_out, logit=args.logit_transform)
                    
            #if args.variables[0] in ["pr", "sfcWind"] and args.sqrt_transform_out:
            #   # visual eval, not normalised but sqrt
            #    visual_sample(model, x_tr_eval, y_tr_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_tr_raw_sqrt', norm_method=args.norm_method_output, norm_stats=norm_stats,
            #                 square_data=False, sqrt_transform=args.sqrt_transform_out)
            #    visual_sample(model, x_te_eval, y_te_eval, save_dir=save_dir + f'img_{epoch_idx + 1}_te_raw_sqrt', norm_method=args.norm_method_output, norm_stats=norm_stats,
            #                 square_data=False, sqrt_transform=args.sqrt_transform_out)
            
            losses_to_img(save_dir, "log.txt", "")
            losses_to_img(save_dir, "log_raw.txt", "raw")
            
        if epoch_idx == 0 or (epoch_idx + 1) % args.save_model_every == 0:# and i >= 30:
            torch.save(model.module.state_dict(), save_dir + f"model_{epoch_idx}.pt")
            
    # Clean up memory
    torch.cuda.empty_cache()