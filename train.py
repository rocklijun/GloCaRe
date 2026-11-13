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
from util.utils import AverageMeter, count_params, init_log, DiceLoss, DiceLoss_sample_weight
from util.dist_helper import setup_distributed
import torch.nn.functional as F

import torch.distributed as dist
import random
import numpy as np

from utils_calc_metrics import evaluate_medical_v2_visualization


def set_seed(seed, rank=0):
    seed = int(seed) + int(rank)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_weighted_segmentation_loss(pred_x, mask_x, weight1_x, criterion_ce, criterion_dice):
    """Image-level: weight1_x is [B] or [B, 1]"""
    # CrossEntropyLoss
    loss_ce_per_pixel = criterion_ce(pred_x, mask_x)  # [B, H, W]
    loss_ce_per_sample = loss_ce_per_pixel.mean(dim=[1, 2])  # [B]

    weight1_x = weight1_x.view(-1)  # [B, 1] -> [B]
    loss_ce_weighted = (loss_ce_per_sample * weight1_x).sum() / (weight1_x.sum() + 1e-8)
    loss_ce_weighted = loss_ce_weighted.view(1)

    # DiceLoss
    loss_dice_weighted = criterion_dice(
        pred_x.softmax(dim=1),
        mask_x.unsqueeze(1).float(),
        sample_weight=weight1_x
    )
    return (loss_ce_weighted + loss_dice_weighted) / 2.0


def compute_pixel_weighted_loss(pred_x, mask_x, weight_map, criterion_ce, criterion_dice):
    loss_ce = criterion_ce(pred_x, mask_x)  # [B, H, W]
    loss_ce_weighted = (loss_ce * weight_map).sum() / (weight_map.sum() + 1e-8)

    pred_softmax = pred_x.softmax(dim=1)
    mask_onehot = F.one_hot(mask_x, num_classes=pred_x.shape[1]).permute(0, 3, 1, 2).float()

    weight_map_exp = weight_map.unsqueeze(1)
    intersect = (pred_softmax * mask_onehot * weight_map_exp).sum(dim=(2, 3))
    union = ((pred_softmax + mask_onehot) * weight_map_exp).sum(dim=(2, 3))
    dice_score = (2.0 * intersect + 1e-5) / (union + 1e-5)

    loss_dice_weighted = 1.0 - dice_score.mean()

    return (loss_ce_weighted + loss_dice_weighted) / 2.0



def Standard_Loss(pred_x, mask_x,  criterion_l, criterion_dice):
    loss_ce = criterion_l(pred_x, mask_x)
    loss_dice = criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())

    if loss_ce.ndim > 0: loss_ce = loss_ce.mean()
    if loss_dice.ndim > 0: loss_dice = loss_dice.mean()

    loss_x = (loss_ce + loss_dice) / 2.0
    return loss_x

from ops.config import parser
def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    seed = cfg.get("seed", 1234)
    set_seed(seed, rank)

    g = torch.Generator()
    g.manual_seed(seed)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)


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

    state_dict = torch.load(f'../pretrained/{cfg["backbone"]}.pth')

    model_dict = model.backbone.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.backbone.load_state_dict(model_dict)

    if cfg['lock_backbone']:
        model.lock_backbone()

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [param for name, param in model.named_parameters()
                   if 'backbone' not in name and 'lswd_module' not in name]
    lswd_params = [param for name, param in model.named_parameters()
                   if 'lswd_module' in name]


    optimizer = AdamW(
        [
            {'params': backbone_params, 'lr': cfg['lr']},
            {'params': head_params, 'lr': cfg['lr'] * cfg['lr_multi']},
            {'params': lswd_params, 'lr': cfg['lr']}  # 通常它的 LR 可以稍小或保持一致
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    warmup_epoch = cfg['warmup_epoch']

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

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    criterion_dice_sw = DiceLoss_sample_weight(n_classes=cfg['nclass']) 

    DATASET_FACTORY = {
        "acdc": ACDCDataset,
        "kvasir": KvasirDataset,
        "cvc": KvasirDataset,
        "synapse": ACDCDataset,
        # ……
    }


    dataset_name = cfg['dataset'].lower()
    DatasetClass = DATASET_FACTORY.get(dataset_name, ACDCDataset) 

    if DatasetClass is None:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    trainset_u = DatasetClass(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = DatasetClass(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = DatasetClass(
        cfg['dataset'], cfg['data_root'], 'val'
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l,
        worker_init_fn=seed_worker,
        generator=g, 

    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u,
        worker_init_fn=seed_worker,
        generator=g, 

    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler,
        worker_init_fn=seed_worker,
        generator=g,

    )

    total_iters = len(trainloader_u) * cfg['epochs'] if len(trainloader_u) > 0 else 1000
    epoch = -1
    warmup_epoch = cfg['warmup_epoch']
    flag_level = cfg.get('flag_level', 'image')

    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema,
                                                        best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)


        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()

            with torch.no_grad():
                model.eval()
                res_u_w = model(img_u_w)
                pred_u_w_eval = res_u_w['pred']
                conf_u_w = pred_u_w_eval.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w_eval.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            res_x = model(
                img_x, epoch=epoch, args=args, iteration=i,
                calc_weight=True,
                need_feat=True,
                flag_level=flag_level
            )
            pred_x = res_x['pred']
            weights_x = res_x.get('weights')
            feats_x = res_x.get('feats')
            fused_map_x = res_x.get('fused_map') 
            w_x_small = res_x.get('weights_pixel_small')

            res_u_s = model(
                torch.cat((img_u_s1, img_u_s2)), epoch=epoch, args=args, iteration=i, comp_drop=True,
                calc_weight=False,
                need_feat=cfg.get('ablation_ind_u_s', False),
                flag_level=flag_level
            )
            pred_u_s_cat = res_u_s['pred']
            pred_u_s1, pred_u_s2 = pred_u_s_cat.chunk(2)

            fused_map_u_s_cat = res_u_s.get('fused_map') 
            fused_map_u_s1, fused_map_u_s2 = None, None
            if fused_map_u_s_cat is not None:
                fused_map_u_s1, fused_map_u_s2 = fused_map_u_s_cat.chunk(2)

            w_s_cat_small = res_u_s.get('weights_pixel_small')
            w_s1_small, w_s2_small = None, None
            if w_s_cat_small is not None:
                w_s1_small, w_s2_small = w_s_cat_small.chunk(2)


            feats_u_w = None
            w_u_w_small = None
            if cfg.get('ablation_ind_u_w', False):
                res_u_w_train = model(img_u_w, calc_weight=True, need_feat=True, flag_level=flag_level)
                feats_u_w = res_u_w_train['feats']
                w_u_w_small = res_u_w_train.get('weights_pixel_small')


            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]

            if flag_level == 'image':
                loss_x = compute_weighted_segmentation_loss(
                    pred_x, mask_x, weights_x, criterion_l, criterion_dice_sw
                )
            elif flag_level == 'pixel':
                loss_x = Standard_Loss( pred_x, mask_x, criterion_l,  criterion_dice)

            elif flag_level == 'both':
                w_img = weights_x['image']
                w_pix = weights_x['pixel']

                loss_x_img = compute_weighted_segmentation_loss(
                    pred_x, mask_x, w_img, criterion_l, criterion_dice_sw
                )
                loss_x_pix = Standard_Loss(pred_x, mask_x, criterion_l,  criterion_dice)

                loss_x = (loss_x_img + loss_x_pix) / 2.0
            else:
                loss_x = Standard_Loss( pred_x, mask_x, criterion_l,  criterion_dice)

            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            loss = (loss_x + loss_u_s) / 2.0

            loss_ind_val = 0.0

            if (epoch >= warmup_epoch) and (flag_level in ['pixel', 'both']):

                mask_high_conf = (conf_u_w >= cfg['conf_thresh'])

                lswd_mod = model.module.lswd_module

                if cfg.get('ablation_ind_x', False):
                    if w_x_small is not None and fused_map_x is not None:
                        loss_ind_val += lswd_mod.compute_loss_selfpace(fused_map_x, w_x_small)
                    else:
                        print('[Ablation: Independent Loss] No fused_map_x or w_x_small.')

                if cfg.get('ablation_ind_u_w', False) and feats_u_w is not None:
                    if w_u_w_small is not None:
                        loss_ind_val += lswd_mod.compute_loss_selfpace(feats_u_w, w_u_w_small,
                                                                       valid_mask=mask_high_conf)
                    else:
                        print('[Ablation: Independent Loss] No w_u_w_small.')

                if cfg.get('ablation_ind_u_s', False):
                    if w_s1_small is not None:
                        loss_ind_val += (
                                                lswd_mod.compute_loss_selfpace(fused_map_u_s1, w_s1_small,
                                                                               valid_mask=mask_high_conf) +
                                                lswd_mod.compute_loss_selfpace(fused_map_u_s2, w_s2_small,
                                                                               valid_mask=mask_high_conf)
                                        ) / 2.0
                    else:
                        print('[Ablation: Independent Loss] No w_s1_small or w_s2_small.')

                ramp_val = min(1.0, (epoch - warmup_epoch) / 5.0)
                loss = loss + (cfg['lambda_ind'] * ramp_val) * loss_ind_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum().item() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))

        mean_dice, dice_class = evaluate_medical_v2_visualization(model, valloader, 'val', cfg)
        mean_dice_ema, dice_class_ema = evaluate_medical_v2_visualization(model_ema, valloader, 'val', cfg)

        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice, dice_class_ema[cls_idx]))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}, EMA: {:.2f}\n'.format(mean_dice, mean_dice_ema))

            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            writer.add_scalar('eval/MeanDice_ema', mean_dice_ema, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)
                writer.add_scalar('eval/%s_dice_ema' % (CLASSES[cfg['dataset']][i]), dice_class_ema[i], epoch)

        is_best = mean_dice > previous_best

        previous_best = max(mean_dice, previous_best)
        previous_best_ema = max(mean_dice_ema, previous_best_ema)

        if mean_dice == previous_best:
            best_epoch = epoch
        if mean_dice_ema == previous_best_ema:
            best_epoch_ema = epoch

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


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
