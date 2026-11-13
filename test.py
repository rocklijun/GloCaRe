import argparse
import time
from copy import deepcopy
import logging
import os
import pprint
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.acdc import ACDCDataset, KvasirDataset
from model.semseg.dpt import DPT

from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed
import torch.nn.functional as F

import torch.distributed as dist

import SimpleITK as sitk
import numpy as np

from utils_calc_metrics import evaluate_medical_v2_visualization


from ops.config import parser
def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        # writer = SummaryWriter(args.save_path)
        # os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Build model
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass'],
        'args': args,
    })


    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

    else:
        raise ValueError("No checkpoint found")

    # Dataset and DataLoader
    # 1. 建立映射关系
    DATASET_FACTORY = {
        "acdc": ACDCDataset,
        "kvasir": KvasirDataset,
        "cvc": KvasirDataset,
        "synapse": ACDCDataset,
        # ……
    }

    # 2. 自动根据 cfg['dataset'] 获取类
    dataset_name = cfg['dataset'].lower()
    DatasetClass = DATASET_FACTORY.get(dataset_name, ACDCDataset)   # # 默认 ACDCDataset。Refer to 9_convert_v3_crop_dataset.py

    if DatasetClass is None:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    # 3. 创建数据集实例
    mode = args.mode
    testset = DatasetClass(cfg['dataset'], cfg['data_root'], mode)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=testsampler)

    eval_outputs = evaluate_medical_v2_visualization(
        model=model,
        loader=testloader,
        mode=mode,
        cfg=cfg,
        save_vis_dir=args.save_vis_dir,
        metrics=("dice", "jaccard", "hd95", "asd"),
        include_background=False,
        return_dict=True,
        debug_export_path=os.environ.get("ACDC_METRIC_DEBUG_EXPORT"),
    )

    results = normalize_eval_results(eval_outputs)

    log_eval_results(
        results=results,
        logger=logger,
        mode=mode,
        cfg=cfg,
        rank=rank,
        classes=CLASSES
    )

def normalize_eval_results(eval_outputs):
    """
    将 evaluate_medical_v2_visualization 的返回值统一转成 dict 格式。

    支持两种输入：
    1. return_dict=False:
       eval_outputs = (mean_dice, dice_class)

    2. return_dict=True:
       eval_outputs = {
           "Dice": {"mean": ..., "class": [...]},
           "Jaccard": {"mean": ..., "class": [...]},
           "HD95": {"mean": ..., "class": [...]},
           "ASD": {"mean": ..., "class": [...]},
       }
    """
    if isinstance(eval_outputs, dict):
        return eval_outputs

    if isinstance(eval_outputs, (tuple, list)) and len(eval_outputs) == 2:
        mean_dice, dice_class = eval_outputs
        return {
            "Dice": {
                "mean": mean_dice,
                "class": dice_class
            }
        }

    raise TypeError(
        "Unsupported evaluation output type: {}. "
        "Expected dict or tuple/list with length 2.".format(type(eval_outputs))
    )


def log_eval_results(results, logger, mode, cfg, rank=0, classes=None):
    """
    统一打印 Dice / Jaccard / HD95 / ASD。
    """
    if rank != 0:
        return

    if classes is None:
        classes = CLASSES

    dataset_name = cfg["dataset"]

    for metric_name, metric_value in results.items():
        metric_mean = metric_value.get("mean", None)
        metric_class = metric_value.get("class", [])

        for cls_idx, value in enumerate(metric_class):
            try:
                cls_name = classes[dataset_name][cls_idx]
            except Exception:
                cls_name = "class_{}".format(cls_idx)

            logger.info(
                "***** Evaluation for dataset {} ***** >>>> Class [{} {}] {}: {:.2f}".format(
                    mode,
                    cls_idx,
                    cls_name,
                    metric_name,
                    value
                )
            )

        if metric_mean is not None:
            logger.info(
                "***** Evaluation for dataset {} ***** >>>> Mean{}: {:.2f}\n".format(
                    mode,
                    metric_name,
                    metric_mean
                )
            )

def get_total_time(start_time, end_time):
    total_seconds = end_time - start_time
    day = total_seconds // 86400
    hour = (total_seconds % 86400) // 3600
    minute = ((total_seconds % 86400) % 3600) // 60
    second = ((total_seconds % 86400) % 3600) % 60
    print('Total time: {}day {}hour {}min {:.2f}s'.format(day, hour, minute, second))

if __name__ == '__main__':
    start = time.time()
    # main()
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    end = time.time()
    get_total_time(
        start, end
    )
