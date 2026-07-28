import os
import time
import random
import argparse

import torch

from modules_loc_variant_temporal import RectUpsampleWithResidualsTemporal
from loss_func import energy_loss_two_sample, energy_loss_multivariate_summed, norm_loss_multivariate_summed
from data import get_data
from load_config import load_config
from utils import (
    make_folder,
    write_config_to_file,
    get_rcm_gcm_combinations,
    get_run_index_from_onehot,
    get_mode_from_kernel_size,
    load_norm_stats_for_variables,
    unnormalise,
    visual_sample,
    get_eval_samples,
    losses_to_img,
    plot_rh,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def build_save_dir(cfg, variables_str, subfolder):
    if cfg.general.server == "ada":
        return (
            f"results/super_temporal/{subfolder}/var-{variables_str}/"
            f"loc-specific-layers_norm-out-{cfg.data.preprocessing.norm_method_output}{cfg.general.save_name}/"
        )
    if cfg.general.server == "euler":
        return (
            "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/eng-results/"
            f"super_temporal/{subfolder}/var-{variables_str}/"
            f"loc-specific-layers_norm-out-{cfg.data.preprocessing.norm_method_output}{cfg.general.save_name}/"
        )
    raise ValueError(f"Unknown server: {cfg.general.server}")


def open_log_file(path, resume_epoch):
    return open(path, "at" if resume_epoch > 0 else "wt")


def get_subfolder(cfg):
    if cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 1:
        return "all"
    if cfg.data.kernel_size_lr == 4 and cfg.data.kernel_size_hr == 1:
        return "super"
    if cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 4:
        return "coarse"
    return f"lr{cfg.data.kernel_size_lr}_hr{cfg.data.kernel_size_hr}"


def infer_cls_ids(x, one_hot_dim, cfg, gcm_list, rcm_list, gcm_dict, rcm_dict, root):
    if one_hot_dim == 0:
        return None
    one_hot = x[:, -one_hot_dim:]
    if cfg.data.ignore_one_hot_gcm:
        return torch.argmax(one_hot, dim=1)
    cls_ids_np = get_run_index_from_onehot(
        one_hot,
        gcm_dict=gcm_dict,
        rcm_dict=rcm_dict,
        gcm_list=gcm_list,
        rcm_list=rcm_list,
        root=root,
    )
    return torch.from_numpy(cls_ids_np).to(x.device)


def compute_raw_loss(y, gen1, gen2, cfg, mode_unnorm, norm_stats):
    if cfg.data.kernel_size_hr != 1:
        return 0.0, 0.0, 0.0

    loss_raw_total = 0.0
    s1_raw_total = 0.0
    s2_raw_total = 0.0

    for i, var in enumerate(cfg.data.variables):
        if len(y.shape) == 3:
            y_var = y[:, i, :]
            gen1_var = gen1[:, i, :]
            gen2_var = gen2[:, i, :]
        else:
            dim_per_var = y.size(1) // len(cfg.data.variables)
            y_var = y[:, i * dim_per_var:(i + 1) * dim_per_var]
            gen1_var = gen1[:, i * dim_per_var:(i + 1) * dim_per_var]
            gen2_var = gen2[:, i * dim_per_var:(i + 1) * dim_per_var]

        s1, s2 = (128, 128) if mode_unnorm == "hr" else (128 // cfg.data.kernel_size_hr, 128 // cfg.data.kernel_size_hr)
        y_raw = unnormalise(y_var, var=var, mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[var])
        gen1_raw = unnormalise(gen1_var, var=var, mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[var])
        gen2_raw = unnormalise(gen2_var, var=var, mode=mode_unnorm, s1=s1, s2=s2, cfg=cfg.data.preprocessing, norm_stats=norm_stats[var])

        loss_raw, s1_raw, s2_raw = energy_loss_two_sample(y_raw, gen1_raw, gen2_raw, verbose=True, beta=cfg.loss.beta)
        loss_raw_total += loss_raw.item()
        s1_raw_total += s1_raw.item()
        s2_raw_total += s2_raw.item()

    return loss_raw_total, s1_raw_total, s2_raw_total


if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        return parser.parse_args()

    args = parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    device = torch.device("cuda")

    if not cfg.model.nicolai_layers:
        raise NotImplementedError("Temporal v2 is only implemented for nicolai layers.")

    if cfg.sparse_layers.double_linear:
        raise NotImplementedError(
            "double_linear=True requires LocalResiduals2, which is not enabled in modules_loc_variant_temporal.py"
        )

    cfg.data.return_timepair = True

    variables_str = "_".join(cfg.data.variables)
    subfolder = get_subfolder(cfg)
    save_dir = build_save_dir(cfg, variables_str, subfolder)
    make_folder(save_dir)
    write_config_to_file(cfg, save_dir)

    log_file = open_log_file(os.path.join(save_dir, "log.txt"), cfg.general.resume_epoch)
    log_file_super = open_log_file(os.path.join(save_dir, "log_super.txt"), cfg.general.resume_epoch)
    log_file_raw = open_log_file(os.path.join(save_dir, "log_raw.txt"), cfg.general.resume_epoch)
    log_file_stats = open_log_file(os.path.join(save_dir, "log_stats.txt"), cfg.general.resume_epoch)
    log_file_time = open_log_file(os.path.join(save_dir, "log_time.txt"), cfg.general.resume_epoch)

    if cfg.general.server == "ada":
        root = "/r/scratch/groups/nm/downscaling/cordex-ALPS-allyear"
    elif cfg.general.server == "euler":
        root = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear"
    else:
        raise ValueError(f"Unknown server: {cfg.general.server}")

    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)

    train_loader, test_loader_in = get_data(cfg, test_size=0.1, shuffle=True)
    print("#training batches:", len(train_loader))

    x_tr_eval_prev, xc_tr_eval_prev, y_tr_eval_prev, x_tr_eval, xc_tr_eval, y_tr_eval = next(iter(train_loader))
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

    if cfg.model.one_hot_in_super and not cfg.data.ignore_one_hot_gcm:
        one_hot_dim = 7
        num_classes = 8
        num_classes_resid = 1 if cfg.model.one_hot_only_in_ups else 8
    elif cfg.model.one_hot_in_super and cfg.data.ignore_one_hot_gcm:
        one_hot_dim = 4
        num_classes = 4
        num_classes_resid = 1 if cfg.model.one_hot_only_in_ups else 4
    else:
        one_hot_dim = 0
        num_classes = 1
        num_classes_resid = 1

    model = RectUpsampleWithResidualsTemporal(
        128 // cfg.data.kernel_size_lr,
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
        softmax=False,
        split_residuals=not cfg.sparse_layers.not_split_residuals,
    ).to(device)

    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    print(f"Built a model with #params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if cfg.general.resume_epoch > 0:
        ckpt = os.path.join(save_dir, f"model_{cfg.general.resume_epoch}.pt")
        model.module.load_state_dict(torch.load(ckpt))

    mode_unnorm = get_mode_from_kernel_size(cfg.data.kernel_size_hr)
    if cfg.data.kernel_size_hr == 1:
        norm_stats = load_norm_stats_for_variables(cfg.data.preprocessing, "hr", cfg.data.variables, device=device)
    else:
        norm_stats = {v: None for v in cfg.data.variables}

    for epoch_idx in range(cfg.general.resume_epoch, cfg.training.num_epochs):
        start_epoch = time.time()
        model.train()

        loss_tr = 0.0
        s1_tr = 0.0
        s2_tr = 0.0
        loss_tr_sup = 0.0
        s1_tr_sup = 0.0
        s2_tr_sup = 0.0
        loss_tr_locs = 0.0
        loss_tr_batchel = 0.0
        n_batches = 0

        loss_tr_raw = 0.0
        s1_tr_raw = 0.0
        s2_tr_raw = 0.0
        n_batches_raw = 0

        for data_batch in train_loader:
            optimizer.zero_grad()

            x_prev, xc_prev, y_prev, x, xc, y = data_batch
            x_prev = x_prev.to(device)
            xc_prev = xc_prev.to(device)
            y_prev = y_prev.to(device)
            x = x.to(device)
            xc = xc.to(device)
            y = y.to(device)

            if len(y.shape) == 3:
                y = y.view(y.shape[0], -1)
            if len(y_prev.shape) == 3:
                y_prev = y_prev.view(y_prev.shape[0], -1)
            if len(xc.shape) == 3:
                xc = xc.view(xc.shape[0], -1)

            cls_ids = infer_cls_ids(x, one_hot_dim, cfg, gcm_list, rcm_list, gcm_dict, rcm_dict, root) if cfg.model.one_hot_in_super else None

            if cfg.sparse_layers.add_intermediate_loss and cfg.sparse_layers.double_linear:
                gen1, _, y_interm1 = model(xc, cls_ids=cls_ids, return_latent=True, y_prev=y_prev)
                gen2, _, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True, y_prev=y_prev)
            elif cfg.sparse_layers.add_intermediate_loss:
                gen1, y_interm1 = model(xc, cls_ids=cls_ids, return_latent=True, y_prev=y_prev)
                gen2, y_interm2 = model(xc, cls_ids=cls_ids, return_latent=True, y_prev=y_prev)
            else:
                gen1 = model(xc, cls_ids=cls_ids, y_prev=y_prev)
                gen2 = model(xc, cls_ids=cls_ids, y_prev=y_prev)

            losses = energy_loss_two_sample(y, gen1, gen2, verbose=True, beta=cfg.loss.beta, patch_size=None)
            if len(cfg.data.variables) > 1:
                losses_per_var = energy_loss_multivariate_summed(y, gen1, gen2, verbose=True, beta=cfg.loss.beta, n_vars=len(cfg.data.variables))
                loss = losses[0] + losses_per_var[0]
            else:
                losses_per_var = None
                loss = losses[0]

            if cfg.sparse_layers.add_intermediate_loss:
                losses_interm = energy_loss_two_sample(y, y_interm1, y_interm2, verbose=True, beta=cfg.loss.beta, patch_size=None)
                loss += losses_interm[0]
            else:
                losses_interm = None

            if cfg.loss.p_norm_loss_loc:
                lossnp, lossnn = norm_loss_multivariate_summed(
                    y,
                    gen1,
                    gen2,
                    cfg.loss.p_norm_loss_loc,
                    beta_norm_loss=cfg.loss.beta_norm_loss,
                    type="loc",
                    agg_norm_loss="mean",
                    n_vars=len(cfg.data.variables),
                )
                loss_tr_locs += lossnp.item()
                if cfg.loss.norm_loss_loc:
                    loss += cfg.loss.lambda_norm_loss_loc * (lossnp + lossnn)

            if cfg.loss.p_norm_loss_batch:
                lossrn, lossrp = norm_loss_multivariate_summed(
                    y,
                    gen1,
                    gen2,
                    cfg.loss.p_norm_loss_batch,
                    beta_norm_loss=cfg.loss.beta_norm_loss,
                    type="batch",
                    agg_norm_loss=cfg.loss.agg_norm_loss,
                    n_vars=len(cfg.data.variables),
                )
                loss_tr_batchel += lossrp.item()
                if cfg.loss.norm_loss_batch:
                    loss += lossrp + lossrn

            loss.backward()
            optimizer.step()

            n_batches += 1
            loss_tr += loss.item()
            loss_tr_sup += losses[0].item()
            s1_tr_sup += losses[1].item()
            s2_tr_sup += losses[2].item()

            if losses_interm is not None and losses_per_var is not None:
                s1_tr += losses[1].item() + losses_per_var[1].item() + losses_interm[1].item()
                s2_tr += losses[2].item() + losses_per_var[2].item() + losses_interm[2].item()
            elif losses_per_var is not None:
                s1_tr += losses[1].item() + losses_per_var[1].item()
                s2_tr += losses[2].item() + losses_per_var[2].item()
            else:
                s1_tr += losses[1].item()
                s2_tr += losses[2].item()

            if cfg.loss.calc_raw_loss and cfg.data.kernel_size_hr == 1:
                if epoch_idx == 0 or ((epoch_idx + 1) % (cfg.general.print_every_nepoch * 10) == 0):
                    if n_batches_raw < 3:
                        n_batches_raw += 1
                        with torch.no_grad():
                            lraw, s1raw, s2raw = compute_raw_loss(y, gen1, gen2, cfg, mode_unnorm, norm_stats)
                            loss_tr_raw += lraw
                            s1_tr_raw += s1raw
                            s2_tr_raw += s2raw

        epoch_time = time.time() - start_epoch
        log_file_time.write(f"epoch took: {epoch_time}\n")
        log_file_time.flush()

        if epoch_idx == 0 or ((epoch_idx + 1) % cfg.general.print_every_nepoch == 0):
            log_full = (
                f"Train [Epoch {epoch_idx + 1}] \tloss: {loss_tr / n_batches:.4f}, "
                f"s1: {s1_tr / n_batches:.4f}, s2: {s2_tr / n_batches:.4f}"
            )
            log_super = (
                f"Train-sup [Epoch {epoch_idx + 1}] \tloss: {loss_tr_sup / n_batches:.4f}, "
                f"s1: {s1_tr_sup / n_batches:.4f}, s2: {s2_tr_sup / n_batches:.4f}"
            )
            log_stats = (
                f"Train-stats [Epoch {epoch_idx + 1}] \tloss_locs: {loss_tr_locs / n_batches:.4f}, "
                f"loss_batchel: {loss_tr_batchel / n_batches:.4f}"
            )

            print(log_full)
            print(log_super)
            log_file.write(log_full + "\n")
            log_file.flush()
            log_file_super.write(log_super + "\n")
            log_file_super.flush()
            log_file_stats.write(log_stats + "\n")
            log_file_stats.flush()

            if n_batches_raw > 0:
                log_raw = (
                    f"Train-raw [Epoch {epoch_idx + 1}] \tloss: {loss_tr_raw / n_batches_raw:.4f}, "
                    f"s1: {s1_tr_raw / n_batches_raw:.4f}, s2: {s2_tr_raw / n_batches_raw:.4f}"
                )
                print(log_raw)
                log_file_raw.write(log_raw + "\n")
                log_file_raw.flush()

        if epoch_idx == 0 or ((epoch_idx + 1) % cfg.general.sample_every_nepoch == 0):
            visual_sample(
                model,
                xc_tr_eval,
                y_tr_eval,
                save_dir=save_dir + f"img_{epoch_idx + 1}_tr_loss-scale_super",
                cfg=cfg,
                norm_stats=None,
                mode_unnorm=mode_unnorm,
                one_hot_dim=one_hot_dim,
                one_hot_in_super=cfg.model.one_hot_in_super,
                conv=False,
                x_one_hot=x_tr_eval,
                gcm_dict=gcm_dict,
                rcm_dict=rcm_dict,
                gcm_list=gcm_list,
                rcm_list=rcm_list,
                y_prev=y_tr_eval_prev,
            )
            visual_sample(
                model,
                xc_te_eval,
                y_te_eval,
                save_dir=save_dir + f"img_{epoch_idx + 1}_te_loss-scale_super",
                cfg=cfg,
                norm_stats=None,
                mode_unnorm=mode_unnorm,
                one_hot_dim=one_hot_dim,
                one_hot_in_super=cfg.model.one_hot_in_super,
                conv=False,
                x_one_hot=x_te_eval,
                gcm_dict=gcm_dict,
                rcm_dict=rcm_dict,
                gcm_list=gcm_list,
                rcm_list=rcm_list,
                y_prev=y_te_eval_prev,
            )

            if cfg.data.kernel_size_hr == 1:
                visual_sample(
                    model,
                    xc_tr_eval,
                    y_tr_eval,
                    save_dir=save_dir + f"img_{epoch_idx + 1}_tr_super",
                    cfg=cfg,
                    norm_stats=norm_stats,
                    mode_unnorm=mode_unnorm,
                    one_hot_dim=one_hot_dim,
                    one_hot_in_super=cfg.model.one_hot_in_super,
                    conv=False,
                    x_one_hot=x_tr_eval,
                    gcm_dict=gcm_dict,
                    rcm_dict=rcm_dict,
                    gcm_list=gcm_list,
                    rcm_list=rcm_list,
                    y_prev=y_tr_eval_prev,
                )
                visual_sample(
                    model,
                    xc_te_eval,
                    y_te_eval,
                    save_dir=save_dir + f"img_{epoch_idx + 1}_te_super",
                    cfg=cfg,
                    norm_stats=norm_stats,
                    mode_unnorm=mode_unnorm,
                    one_hot_dim=one_hot_dim,
                    one_hot_in_super=cfg.model.one_hot_in_super,
                    conv=False,
                    x_one_hot=x_te_eval,
                    gcm_dict=gcm_dict,
                    rcm_dict=rcm_dict,
                    gcm_list=gcm_list,
                    rcm_list=rcm_list,
                    y_prev=y_te_eval_prev,
                )

            losses_to_img(save_dir, "log.txt", "", "_full")
            losses_to_img(save_dir, "log_super.txt", "sup", "_super")
            losses_to_img(save_dir, "log_raw.txt", "raw", "_raw")

            trues, samples = get_eval_samples(
                model.module,
                test_loader_in,
                cfg,
                device,
                mode_unnorm=mode_unnorm,
                norm_stats=None,
                input_mode="xc",
                output_mode="y",
                one_hot_in_super=cfg.model.one_hot_in_super,
                conv=False,
                one_hot_dim=one_hot_dim,
                gcm_dict=gcm_dict,
                rcm_dict=rcm_dict,
                gcm_list=gcm_list,
                rcm_list=rcm_list,
                temporal=True,
            )
            for i, var in enumerate(cfg.data.variables):
                plot_rh(
                    trues[:, i, :, :],
                    samples[:, i, :, :, :],
                    epoch_idx,
                    save_dir,
                    file_suffix=f"_temporal-var-{var}",
                    include_quantile=True,
                )


        if epoch_idx == 0 or ((epoch_idx + 1) % cfg.training.save_model_every == 0):
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_{epoch_idx}.pt"))

    torch.cuda.empty_cache()
