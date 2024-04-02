import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, DistributedSampler

import cora
from cora.engine import evaluate
from cora.datasets import build_dataset, get_coco_api_from_dataset
from cora.models import build_model
from cora.util.misc import (
    collate_fn,
    fast_ema,
    get_rank,
    init_distributed_mode,
    save_on_master,
)

if __name__ == "__main__":
    config_file = "R50x4_dab_ovd_3enc_apm128_splcls0.2_relabel_noinit.py"
    region_prompt = "region_prompt_R50x4.pth"
    output_dir = "R50x4_dab_ovd_3enc_apm128_splcls0.2_relabel_noinit"
    checkpoint_file = "COCO_RN50x4.pth.pth_one_epoch_after"

    args = Config.fromfile(
        os.path.join(os.path.dirname(cora.__file__), "..", "configs/COCO", config_file)
    )
    args.resume = os.path.join(
        os.path.dirname(cora.__file__), "..", "logs", checkpoint_file
    )
    args.region_prompt_path = os.path.join(
        os.path.dirname(cora.__file__), "..", "logs", region_prompt
    )
    args.coco_path = os.path.join(os.path.dirname(cora.__file__), "..", "data/coco")
    args.output_dir = os.path.join(
        os.path.dirname(cora.__file__), "..", "logs", output_dir
    )

    torch.backends.cuda.enable_mem_efficient_sdp(args.enable_mem_efficient_sdp)
    torch.backends.cuda.enable_flash_sdp(args.enable_flash_sdp)
    torch.backends.cuda.enable_math_sdp(args.enable_math_sdp)
    init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = fast_ema(
            model,
            decay=args.model_ema_decay,
            device="",
            resume="",
            skip_keywords=["classifier", "backbone"],
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params in model: ", n_parameters)

    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.loda_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)

    if args.resume:
        print(f"loading checkpoint from {args.resume}")
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "best_performance" in checkpoint:
            best_performance = checkpoint["best_performance"]
        if "model_ema" in checkpoint and model_ema is not None:
            mem_file = io.BytesIO()
            torch.save({"state_dict_ema": checkpoint["model_ema"]}, mem_file)
            mem_file.seek(0)
            model_ema._load_checkpoint(mem_file)

    test_stats, coco_evaluator = evaluate(
        model_ema.ema if args.model_ema else model,
        criterion,
        post_processors,
        data_loader_val,
        base_ds,
        device,
        args.output_dir,
        args=args,
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
