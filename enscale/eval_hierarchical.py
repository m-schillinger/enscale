import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_v2 import ConfigV2
from data_v2 import (
    DownscalingDatasetV2,
    PreprocessorV2,
    build_reader_v2,
    build_run_specs_v2,
)
from load_config_v2 import load_config_v2
from modules import HierarchicalWrapper, ModelSpec, StoUNet
from modules_cnn import Generator16x, Generator2x, Generator4x, Generator4xConcat
from modules_loc_variant import RectUpsampleWithResiduals
from utils import (
    build_save_dir,
    get_ensemble_encoding_scheme,
    get_mode_from_kernel_size,
    get_rcm_gcm_combinations,
    get_run_index_from_onehot,
    load_norm_stats_for_variables,
    make_folder,
    unnormalise,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical v2 evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to hierarchical inference config")
    parser.add_argument("--split", type=str, default=None, choices=["test", "inference"])
    parser.add_argument("--submode", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu")
    return parser.parse_args()


def _server_output_root(cfg: ConfigV2) -> str:
    if cfg.general.server == "ada":
        return cfg.general.save_dir_root_local
    if cfg.general.server == "euler":
        return cfg.general.save_dir_root_server
    raise ValueError(f"Unknown server: {cfg.general.server}")


def _infer_super_subfolder(cfg: ConfigV2) -> str:
    if cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 1:
        return "all"
    if cfg.data.kernel_size_lr == 4 and cfg.data.kernel_size_hr == 1:
        return "super"
    if cfg.data.kernel_size_lr == 16 and cfg.data.kernel_size_hr == 4:
        return "coarse"
    return f"lr{cfg.data.kernel_size_lr}_hr{cfg.data.kernel_size_hr}"


def _resolve_stage_checkpoint(stage, stage_cfg: ConfigV2) -> str:
    if stage.checkpoint_source == "pretrained":
        if not stage.checkpoint_path:
            raise ValueError(f"Stage '{stage.name}' is missing checkpoint_path")
        return stage.checkpoint_path

    if stage.train_run_dir:
        ckpt_dir = stage.train_run_dir
    else:
        variables_str = "_".join(stage_cfg.data.variables)
        subfolder = stage.subfolder
        if subfolder is None and stage.stage in {"super", "super_temporal"}:
            subfolder = _infer_super_subfolder(stage_cfg)
        if subfolder is None:
            subfolder = ""
        ckpt_dir = build_save_dir(stage_cfg, variables_str, subfolder=subfolder, stage=stage.stage)

    if stage.checkpoint_file:
        ckpt_name = stage.checkpoint_file
    else:
        ckpt_name = f"model_{stage.epoch}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint for stage '{stage.name}' does not exist: {ckpt_path}"
        )
    return ckpt_path


def _get_one_hot_metadata(cfg: ConfigV2):
    scheme = get_ensemble_encoding_scheme(cfg)
    if scheme is None or cfg.data.type != "cordex_ensemble":
        return scheme, 0, None, None, None, None

    root = cfg.data.cordex.root
    gcm_list, rcm_list, gcm_dict, rcm_dict = get_rcm_gcm_combinations(root)
    if scheme == "gcm":
        one_hot_dim = len(gcm_dict)
    elif scheme == "rcm":
        one_hot_dim = len(rcm_dict)
    else:
        one_hot_dim = len(gcm_dict) + len(rcm_dict)

    return scheme, one_hot_dim, gcm_list, rcm_list, gcm_dict, rcm_dict


def _build_stage_model(
    stage_cfg: ConfigV2,
    stage,
    in_dim: int,
    one_hot_dim: int,
    use_one_hot: bool,
    one_hot_option: str,
    scheme: Optional[str],
    gcm_list,
    gcm_dict,
    rcm_dict,
    device,
):
    n_vars = len(stage_cfg.data.variables)
    out_dim = int(128 / stage_cfg.data.kernel_size_hr) ** 2 * n_vars

    if stage_cfg.model.nicolai_layers:
        if stage_cfg.model.one_hot_in_super:
            if scheme == "gcm":
                num_classes = len(gcm_dict)
            elif scheme == "rcm":
                num_classes = len(rcm_dict)
            elif scheme == "gcm+rcm":
                num_classes = len(gcm_list)
            else:
                num_classes = 1
            if stage_cfg.model.one_hot_only_in_ups:
                num_classes_resid = 1
            else:
                num_classes_resid = num_classes
        else:
            num_classes = 1
            num_classes_resid = 1

        model = RectUpsampleWithResiduals(
            128 // stage_cfg.data.kernel_size_lr,
            128 // stage_cfg.data.kernel_size_hr,
            n_features=n_vars,
            num_classes=num_classes,
            num_classes_resid=num_classes_resid,
            num_neighbors_ups=stage_cfg.sparse_layers.num_neighbors_ups,
            num_neighbors_res=stage_cfg.sparse_layers.num_neighbors_res,
            map_dim=stage_cfg.sparse_layers.latent_dim,
            noise_dim=stage_cfg.model.noise_dim,
            mlp_hidden=stage_cfg.model.hidden_dim,
            mlp_depth=stage_cfg.sparse_layers.mlp_depth,
            noise_dim_mlp=stage_cfg.sparse_layers.noise_dim_mlp,
            double_linear=stage_cfg.sparse_layers.double_linear,
            split_residuals=not stage_cfg.sparse_layers.not_split_residuals,
        ).to(device)
        return model, out_dim

    if stage_cfg.model.conv and stage_cfg.data.kernel_size_lr == 16 and stage_cfg.data.kernel_size_hr == 1:
        model = Generator16x(conv_dim=stage_cfg.model.conv_dim, n_channels=n_vars).to(device)
        return model, out_dim

    if stage_cfg.model.conv and (stage_cfg.data.kernel_size_lr // stage_cfg.data.kernel_size_hr == 4):
        model = Generator4x(
            conv_dim=stage_cfg.model.conv_dim,
            n_channels=n_vars,
            one_hot_channel=use_one_hot,
            one_hot_dim=one_hot_dim,
            image_size=128 // stage_cfg.data.kernel_size_lr,
        ).to(device)
        return model, out_dim

    if stage_cfg.model.conv_concat and (stage_cfg.data.kernel_size_lr // stage_cfg.data.kernel_size_hr == 4):
        model = Generator4xConcat(
            conv_dim=stage_cfg.model.conv_dim,
            n_channels=n_vars,
            one_hot_channel=use_one_hot,
            one_hot_dim=one_hot_dim,
            num_noise_channels=stage_cfg.model.num_noise_channels,
            image_size=128 // stage_cfg.data.kernel_size_lr,
        ).to(device)
        return model, out_dim

    if stage_cfg.model.conv and (stage_cfg.data.kernel_size_lr // stage_cfg.data.kernel_size_hr == 2):
        model = Generator2x(
            conv_dim=stage_cfg.model.conv_dim,
            n_channels=n_vars,
            image_size=128 // stage_cfg.data.kernel_size_lr,
        ).to(device)
        return model, out_dim

    dense_in_dim = in_dim
    if use_one_hot and one_hot_option == "concat":
        dense_in_dim += one_hot_dim
    if stage_cfg.model.add_x_in_super:
        dense_in_dim += 720

    preproc_input_dims = None
    if stage_cfg.model.preproc_layer and stage.stage in {"coarse", "coarse_temporal"}:
        n_vars_lr = len(stage_cfg.data.variables_lr or stage_cfg.data.variables)
        preproc_dims = [720 for _ in range(n_vars_lr)] + [5]
        if one_hot_dim > 0:
            preproc_dims.append(one_hot_dim)
        if stage_cfg.model.method == "eng_temporal":
            interm_dim_per_var = out_dim // len(stage_cfg.data.variables)
            preproc_dims.extend([interm_dim_per_var for _ in stage_cfg.data.variables])
        preproc_input_dims = np.array(preproc_dims)

    model = StoUNet(
        dense_in_dim,
        out_dim,
        stage_cfg.model.num_layer,
        stage_cfg.model.hidden_dim,
        stage_cfg.model.noise_dim,
        add_bn=stage_cfg.model.bn,
        out_act=stage_cfg.model.out_act,
        resblock=stage_cfg.model.mlp,
        noise_std=stage_cfg.model.noise_std,
        preproc_layer=stage_cfg.model.preproc_layer,
        input_dims_for_preproc=preproc_input_dims,
        preproc_dim=stage_cfg.model.preproc_dim,
        layer_shrinkage=stage_cfg.model.layer_shrinkage,
        dropout=stage_cfg.model.dropout,
    ).to(device)
    return model, out_dim


def _build_hierarchical_chain(cfg: ConfigV2, in_dim: int, device):
    if not cfg.inference.stages:
        raise ValueError("No stages configured. Set inference.hierarchical=true and provide inference.stages")

    scheme, one_hot_dim, gcm_list, rcm_list, gcm_dict, rcm_dict = _get_one_hot_metadata(cfg)
    model_specs: List[ModelSpec] = []
    stage_manifests = []

    current_in_dim = in_dim
    for idx, stage in enumerate(cfg.inference.stages):
        stage_cfg = load_config_v2(stage.config_path)

        use_one_hot = bool(stage.use_one_hot)
        one_hot_option = stage.one_hot_option
        if stage_cfg.model.nicolai_layers and use_one_hot and one_hot_option == "concat":
            one_hot_option = "argument"

        model, out_dim = _build_stage_model(
            stage_cfg,
            stage=stage,
            in_dim=current_in_dim,
            one_hot_dim=one_hot_dim,
            use_one_hot=use_one_hot,
            one_hot_option=one_hot_option,
            scheme=scheme,
            gcm_list=gcm_list,
            gcm_dict=gcm_dict,
            rcm_dict=rcm_dict,
            device=device,
        )

        ckpt_path = _resolve_stage_checkpoint(stage, stage_cfg)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        model_specs.append(
            ModelSpec(
                model=model,
                vars_as_channels=bool(stage.vars_as_channels),
                use_one_hot=use_one_hot,
                noise_dim=stage.noise_dim,
                one_hot_option=one_hot_option,
            )
        )

        stage_name = stage.name if stage.name else f"stage_{idx}"
        stage_manifests.append({
            "name": stage_name,
            "stage": stage.stage,
            "config_path": stage.config_path,
            "checkpoint": ckpt_path,
        })

        current_in_dim = out_dim

    chain = HierarchicalWrapper(model_specs, n_vars=len(cfg.data.variables), one_hot_dim=one_hot_dim)
    chain.to(device)
    chain.eval()

    return chain, stage_manifests, (scheme, one_hot_dim, gcm_list, rcm_list, gcm_dict, rcm_dict), current_in_dim


def _build_single_run_loader(cfg: ConfigV2, run, split: str, submode: Optional[str], batch_size: int):
    reader = build_reader_v2(cfg)
    preproc = PreprocessorV2(cfg.data.preprocessing)
    ds = DownscalingDatasetV2(
        reader=reader,
        preproc=preproc,
        run=run,
        variables_lr=cfg.data.variables_lr or cfg.data.variables,
        variables_hr=cfg.data.variables,
        kernel_size_lr=cfg.data.kernel_size_lr,
        kernel_size_hr=cfg.data.kernel_size_hr,
        split=split,
        submode=submode,
        return_timepair=False,
        inference_mode=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )


def _unnormalise_samples(samples: torch.Tensor, cfg: ConfigV2, norm_stats, mode_unnorm: str) -> torch.Tensor:
    # samples: [B, D, S] or [B, V, P, S]
    if samples.dim() == 4:
        bsz, n_vars, pix, n_samples = samples.shape
    elif samples.dim() == 3:
        bsz, dim, n_samples = samples.shape
        n_vars = len(cfg.data.variables)
        pix = dim // n_vars
        samples = samples.view(bsz, n_vars, pix, n_samples)
    else:
        raise ValueError(f"Unexpected sample shape: {tuple(samples.shape)}")

    s1 = int(pix ** 0.5)
    s2 = s1
    out_per_var = []
    for vi, var in enumerate(cfg.data.variables):
        var_samples = []
        for si in range(n_samples):
            stats = norm_stats.get(var) if isinstance(norm_stats, dict) else None
            var_u = unnormalise(
                samples[:, vi, :, si],
                var=var,
                mode=mode_unnorm,
                s1=s1,
                s2=s2,
                cfg=cfg.data.preprocessing,
                norm_stats=stats,
            )
            var_samples.append(var_u)
        out_per_var.append(torch.stack(var_samples, dim=-1))
    return torch.stack(out_per_var, dim=1)


def _compute_cls_ids(
    x,
    scheme,
    one_hot_dim,
    gcm_list,
    rcm_list,
    gcm_dict,
    rcm_dict,
    root,
):
    if one_hot_dim <= 0:
        return None

    one_hot = x[:, -one_hot_dim:]
    if scheme in {"gcm", "rcm"}:
        return torch.argmax(one_hot, dim=1)

    cls_ids_np = get_run_index_from_onehot(
        one_hot,
        gcm_dict=gcm_dict,
        rcm_dict=rcm_dict,
        gcm_list=gcm_list,
        rcm_list=rcm_list,
        root=root,
        mode="joint",
    )
    return torch.from_numpy(cls_ids_np).to(x.device)


def _resolve_selected_run_indices(cfg: ConfigV2) -> List[int]:
    if cfg.data.type != "cordex_ensemble":
        return [0]

    if cfg.data.run_indices is not None:
        return [int(i) for i in cfg.data.run_indices]
    if cfg.data.runs.selection == "first_n":
        return list(range(int(cfg.data.runs.n_models)))
    if cfg.data.runs.selection == "explicit":
        return [int(i) for i in cfg.data.runs.indices]
    raise ValueError(f"Unknown run selection: {cfg.data.runs.selection}")


def main():
    args = parse_args()
    cfg = load_config_v2(args.config)

    if not cfg.inference.hierarchical:
        raise ValueError("Set inference.hierarchical=true and provide inference.stages for this script")

    split = args.split or cfg.inference.split
    submode = args.submode if args.submode is not None else cfg.inference.submode

    if split not in {"test", "inference"}:
        raise ValueError("split must be one of: test, inference")

    batch_size = cfg.inference.batch_size or cfg.training.batch_size
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    runs = build_run_specs_v2(cfg)
    run_indices = _resolve_selected_run_indices(cfg)
    if len(runs) != len(run_indices):
        raise ValueError("Run-spec length mismatch with selected run indices")

    # Peek one batch to infer first-stage input dimensionality.
    warm_loader = _build_single_run_loader(cfg, runs[0], split=split, submode=submode, batch_size=batch_size)
    x0, _, _ = next(iter(warm_loader))
    in_dim = int(x0.shape[1])

    chain, stage_manifest, one_hot_meta, _ = _build_hierarchical_chain(cfg, in_dim=in_dim, device=device)

    scheme, one_hot_dim, gcm_list, rcm_list, gcm_dict, rcm_dict = one_hot_meta
    root_for_runs = cfg.data.cordex.root if cfg.data.type == "cordex_ensemble" else cfg.data.data_dir

    final_stage_cfg = load_config_v2(cfg.inference.stages[-1].config_path)
    mode_unnorm = get_mode_from_kernel_size(final_stage_cfg.data.kernel_size_hr)
    if mode_unnorm == "hr":
        norm_stats = load_norm_stats_for_variables(
            final_stage_cfg.data.preprocessing,
            mode="hr",
            variables=final_stage_cfg.data.variables,
            device=device,
        )
    else:
        norm_stats = {v: None for v in final_stage_cfg.data.variables}

    output_root = os.path.join(
        _server_output_root(cfg),
        cfg.inference.output_subdir,
        split,
        submode or "default",
    )
    make_folder(output_root)

    manifest_path = os.path.join(output_root, "manifest.pt")
    torch.save(
        {
            "config_path": os.path.abspath(args.config),
            "split": split,
            "submode": submode,
            "sample_size": cfg.inference.sample_size,
            "stages": stage_manifest,
            "run_indices": run_indices,
        },
        manifest_path,
    )

    for run_idx, run in zip(run_indices, runs):
        loader = _build_single_run_loader(
            cfg,
            run,
            split=split,
            submode=submode,
            batch_size=batch_size,
        )

        run_samples = []
        for x, _, _ in loader:
            x = x.to(device)
            cls_ids = _compute_cls_ids(
                x,
                scheme,
                one_hot_dim,
                gcm_list,
                rcm_list,
                gcm_dict,
                rcm_dict,
                root_for_runs,
            )
            with torch.no_grad():
                samples = chain.sample(
                    x,
                    sample_size=cfg.inference.sample_size,
                    x_onehot=x,
                    cls_ids=cls_ids,
                )
            samples = _unnormalise_samples(samples, final_stage_cfg, norm_stats, mode_unnorm)
            run_samples.append(samples.cpu())

        out = torch.cat(run_samples, dim=0)
        out_path = os.path.join(output_root, f"run_idx{run_idx}.pt")
        torch.save(out, out_path)


if __name__ == "__main__":
    main()
