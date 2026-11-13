import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure



def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    if _is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def _safe_divide(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return numerator / denominator


def _is_2d_dataset(cfg):
    return cfg.get("dataset", "").lower() in ["kvasir", "cvc", "polyp"]


def _align_pred_and_gt_arrays(pred, gt):
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    if pred.shape == gt.shape:
        return pred, gt

    pred_squeezed = np.squeeze(pred)
    gt_squeezed = np.squeeze(gt)

    if pred_squeezed.shape == gt_squeezed.shape:
        return pred_squeezed, gt_squeezed

    # pred=(B,H,W), gt=(B,1,H,W)
    if gt.ndim == pred.ndim + 1 and gt.shape[1] == 1:
        gt_tmp = np.squeeze(gt, axis=1)
        if gt_tmp.shape == pred.shape:
            return pred, gt_tmp

    # pred=(D,H,W), gt=(1,D,H,W) 或 pred=(H,W), gt=(1,H,W)
    if gt.ndim == pred.ndim + 1 and gt.shape[0] == 1:
        gt_tmp = np.squeeze(gt, axis=0)
        if gt_tmp.shape == pred.shape:
            return pred, gt_tmp

    # pred=(1,D,H,W), gt=(D,H,W) 或 pred=(1,H,W), gt=(H,W)
    if pred.ndim == gt.ndim + 1 and pred.shape[0] == 1:
        pred_tmp = np.squeeze(pred, axis=0)
        if pred_tmp.shape == gt.shape:
            return pred_tmp, gt

    raise ValueError(f"pred and gt must have same shape after alignment, got {pred.shape} vs {gt.shape}")


def _align_pred_and_gt_tensors(pred_mask, mask):
    if pred_mask.shape == mask.shape:
        return pred_mask, mask

    # label 常见格式: (B,1,H,W)，pred: (B,H,W)
    if mask.ndim == pred_mask.ndim + 1 and mask.shape[1] == 1:
        mask_tmp = mask.squeeze(1)
        if mask_tmp.shape == pred_mask.shape:
            return pred_mask, mask_tmp

    # 当前报错对应的情况: pred=(D,H,W), mask=(1,D,H,W)
    if mask.ndim == pred_mask.ndim + 1 and mask.shape[0] == 1:
        mask_tmp = mask.squeeze(0)
        if mask_tmp.shape == pred_mask.shape:
            return pred_mask, mask_tmp

    # pred 可能多一个 batch=1 维
    if pred_mask.ndim == mask.ndim + 1 and pred_mask.shape[0] == 1:
        pred_tmp = pred_mask.squeeze(0)
        if pred_tmp.shape == mask.shape:
            return pred_tmp, mask

    # 最后兜底：仅 squeeze 单例维度，但不破坏非单例维度
    pred_tmp = pred_mask.squeeze()
    mask_tmp = mask.squeeze()
    if pred_tmp.shape == mask_tmp.shape:
        return pred_tmp, mask_tmp

    raise ValueError(
        f"pred_mask and mask cannot be aligned: pred={tuple(pred_mask.shape)}, mask={tuple(mask.shape)}"
    )


def _iter_cases_for_surface_metrics(pred_np, mask_np, cfg):
    pred_np, mask_np = _align_pred_and_gt_arrays(pred_np, mask_np)

    if _is_2d_dataset(cfg):
        # 常规 2D batch: (B,H,W)
        if pred_np.ndim == 2:
            yield pred_np, mask_np
        elif pred_np.ndim == 3:
            for b in range(pred_np.shape[0]):
                yield pred_np[b], mask_np[b]
        else:
            raise ValueError(f"Unsupported 2D pred shape for surface metrics: {pred_np.shape}")
    else:
        # 非 2D 数据默认按照 3D volume 处理
        if pred_np.ndim == 3:
            yield pred_np, mask_np
        elif pred_np.ndim == 4:
            # 若数据确实是 (B,D,H,W)，则逐 volume 处理
            for b in range(pred_np.shape[0]):
                yield pred_np[b], mask_np[b]
        elif pred_np.ndim == 2:
            # 兼容某些非 polyp 但实际为 2D 的数据
            yield pred_np, mask_np
        else:
            raise ValueError(f"Unsupported 3D pred shape for surface metrics: {pred_np.shape}")


def _surface_distances(pred, gt, spacing=None, connectivity=1):
    pred, gt = _align_pred_and_gt_arrays(pred, gt)

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if spacing is None:
        spacing = tuple([1.0] * pred.ndim)

    if len(spacing) != pred.ndim:
        spacing = tuple([1.0] * pred.ndim)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    # 两者都为空：距离为 0
    if pred_sum == 0 and gt_sum == 0:
        return np.array([0.0], dtype=np.float64)

    # 一者为空：HD95 / ASD 无严格定义，返回 None，后续跳过
    if pred_sum == 0 or gt_sum == 0:
        return None

    footprint = generate_binary_structure(pred.ndim, connectivity)

    pred_surface = pred ^ binary_erosion(pred, structure=footprint, iterations=1, border_value=0)
    gt_surface = gt ^ binary_erosion(gt, structure=footprint, iterations=1, border_value=0)

    # 极端情况下 surface 为空，退化为原 mask
    if pred_surface.sum() == 0:
        pred_surface = pred

    if gt_surface.sum() == 0:
        gt_surface = gt

    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    surface_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred]).astype(np.float64)

    return surface_distances


def _compute_hd95_asd(pred, gt, spacing=None):
    surface_distances = _surface_distances(pred, gt, spacing=spacing)

    if surface_distances is None:
        return np.nan, np.nan

    hd95 = np.percentile(surface_distances, 95)
    asd = np.mean(surface_distances)

    return float(hd95), float(asd)


def _prepare_spacing(cfg, ndim):
    spacing = cfg.get("spacing", None)

    if spacing is None:
        return tuple([1.0] * ndim)

    if isinstance(spacing, (int, float)):
        return tuple([float(spacing)] * ndim)

    if len(spacing) != ndim:
        return tuple([1.0] * ndim)

    return tuple([float(s) for s in spacing])


def evaluate_medical_v2_visualization(
    model,
    loader,
    mode,
    cfg,
    save_vis_dir=None,
    metrics=("dice", "jaccard", "hd95", "asd"),
    include_background=False,
    return_dict=False,
    debug_export_path=None,
    return_raw_accumulators=False,
):
    """
    医学图像分割评估函数，支持 Dice / Jaccard / HD95 / ASD。
    """

    model.eval()
    assert mode in ["val", "test", "valtest"]

    metrics = [m.lower() for m in metrics]
    nclass = cfg["nclass"]

    if include_background:
        class_ids = list(range(0, nclass))
    else:
        class_ids = list(range(1, nclass))

    num_eval_classes = len(class_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dice / Jaccard 累计器
    total_inter = torch.zeros(num_eval_classes, device=device)
    total_pred = torch.zeros(num_eval_classes, device=device)
    total_gt = torch.zeros(num_eval_classes, device=device)

    # HD95 / ASD 累计器
    hd95_sum = torch.zeros(num_eval_classes, device=device)
    hd95_count = torch.zeros(num_eval_classes, device=device)

    asd_sum = torch.zeros(num_eval_classes, device=device)
    asd_count = torch.zeros(num_eval_classes, device=device)

    need_save = (save_vis_dir is not None) and (_get_rank() == 0)

    if need_save:
        os.makedirs(save_vis_dir, exist_ok=True)
        print(f"Visualization enabled. Saving to {save_vis_dir}")

    with torch.no_grad():
        for image, label, filename_list in loader:
            case_id = filename_list[0] if isinstance(filename_list, (list, tuple)) else str(filename_list)

            img = image.to(device, non_blocking=True)
            mask = label.to(device, non_blocking=True).long()

            h_orig, w_orig = img.shape[-2:]

            img_input = F.interpolate(
                img,
                size=(cfg["crop_size"], cfg["crop_size"]),
                mode="bilinear",
                align_corners=False
            )

            # 非 2D 数据：原始代码把 volume 的 slice 维度移到 batch 维度
            if not _is_2d_dataset(cfg):
                img_input = img_input.permute(1, 0, 2, 3)

            res_x = model(img_input)
            pred_logits = res_x["pred"]

            pred_logits = F.interpolate(
                pred_logits,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False
            )

            pred_mask = pred_logits.argmax(dim=1).long()

            pred_eval, mask_eval = _align_pred_and_gt_tensors(pred_mask, mask)

            for idx, cls in enumerate(class_ids):
                pred_cls = pred_eval == cls
                gt_cls = mask_eval == cls

                inter = (pred_cls & gt_cls).sum()
                pred_area = pred_cls.sum()
                gt_area = gt_cls.sum()

                total_inter[idx] += inter
                total_pred[idx] += pred_area
                total_gt[idx] += gt_area

            need_surface_metric = ("hd95" in metrics) or ("asd" in metrics)

            if need_surface_metric:
                pred_np = pred_eval.detach().cpu().numpy()
                mask_np = mask_eval.detach().cpu().numpy()

                for pred_case, mask_case in _iter_cases_for_surface_metrics(pred_np, mask_np, cfg):
                    spacing = _prepare_spacing(cfg, pred_case.ndim)

                    for idx, cls in enumerate(class_ids):
                        pred_cls_np = pred_case == cls
                        gt_cls_np = mask_case == cls

                        hd95, asd = _compute_hd95_asd(
                            pred_cls_np,
                            gt_cls_np,
                            spacing=spacing
                        )

                        if not np.isnan(hd95):
                            hd95_sum[idx] += hd95
                            hd95_count[idx] += 1

                        if not np.isnan(asd):
                            asd_sum[idx] += asd
                            asd_count[idx] += 1

            if need_save:
                is_3d_data = (
                    pred_eval.ndim == 3 and not _is_2d_dataset(cfg)
                )

                img_cpu = image[0]
                mask_cpu = mask_eval.detach().cpu().numpy()
                pred_cpu = pred_eval.detach().cpu().numpy()

                if is_3d_data:
                    import SimpleITK as sitk

                    image_np = img_cpu.cpu().numpy().astype(np.float32)

                    image_itk = sitk.GetImageFromArray(image_np)
                    label_itk = sitk.GetImageFromArray(mask_cpu.astype(np.float32))
                    pred_itk = sitk.GetImageFromArray(pred_cpu.astype(np.float32))

                    spacing = _prepare_spacing(cfg, pred_cpu.ndim)
                    if len(spacing) == 3:
                        sitk_spacing = tuple(reversed(spacing))
                    else:
                        sitk_spacing = tuple([1.0] * 3)

                    image_itk.SetSpacing(sitk_spacing)
                    label_itk.SetSpacing(sitk_spacing)
                    pred_itk.SetSpacing(sitk_spacing)

                    sitk.WriteImage(
                        image_itk,
                        os.path.join(save_vis_dir, f"{case_id}_image.nii.gz")
                    )
                    sitk.WriteImage(
                        label_itk,
                        os.path.join(save_vis_dir, f"{case_id}_label.nii.gz")
                    )
                    sitk.WriteImage(
                        pred_itk,
                        os.path.join(save_vis_dir, f"{case_id}_pred.nii.gz")
                    )

                else:
                    img_save = unnormalize(img_cpu)

                    if mask_cpu.ndim == 3:
                        mask_show = mask_cpu[0]
                    else:
                        mask_show = mask_cpu

                    if pred_cpu.ndim == 3:
                        pred_show = pred_cpu[0]
                    else:
                        pred_show = pred_cpu

                    if nclass == 2:
                        label_save = (mask_show * 255).astype(np.uint8)
                        pred_save = (pred_show * 255).astype(np.uint8)
                    else:
                        label_save = mask_show.astype(np.uint8)
                        pred_save = pred_show.astype(np.uint8)

                    Image.fromarray(img_save).save(
                        os.path.join(save_vis_dir, f"{case_id}_image.png")
                    )
                    Image.fromarray(label_save).save(
                        os.path.join(save_vis_dir, f"{case_id}_label.png")
                    )
                    Image.fromarray(pred_save).save(
                        os.path.join(save_vis_dir, f"{case_id}_pred.png")
                    )

    if _is_dist_avail_and_initialized():
        dist.all_reduce(total_inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_gt, op=dist.ReduceOp.SUM)

        dist.all_reduce(hd95_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(hd95_count, op=dist.ReduceOp.SUM)

        dist.all_reduce(asd_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(asd_count, op=dist.ReduceOp.SUM)

    dice_class = []
    jaccard_class = []

    for idx in range(num_eval_classes):
        inter = total_inter[idx].item()
        pred_area = total_pred[idx].item()
        gt_area = total_gt[idx].item()

        dice = _safe_divide(
            2.0 * inter,
            pred_area + gt_area,
            default=0.0
        ) * 100.0

        jaccard = _safe_divide(
            inter,
            pred_area + gt_area - inter,
            default=0.0
        ) * 100.0

        dice_class.append(dice)
        jaccard_class.append(jaccard)

    mean_dice = float(np.mean(dice_class)) if len(dice_class) > 0 else 0.0
    mean_jaccard = float(np.mean(jaccard_class)) if len(jaccard_class) > 0 else 0.0

    hd95_class = []
    asd_class = []

    for idx in range(num_eval_classes):
        if hd95_count[idx].item() > 0:
            hd95_value = hd95_sum[idx].item() / hd95_count[idx].item()
        else:
            hd95_value = np.nan

        if asd_count[idx].item() > 0:
            asd_value = asd_sum[idx].item() / asd_count[idx].item()
        else:
            asd_value = np.nan

        hd95_class.append(hd95_value)
        asd_class.append(asd_value)

    mean_hd95 = float(np.nanmean(hd95_class)) if np.any(~np.isnan(hd95_class)) else np.nan
    mean_asd = float(np.nanmean(asd_class)) if np.any(~np.isnan(asd_class)) else np.nan

    results = {
        "Dice": {
            "mean": mean_dice,
            "class": dice_class
        },
        "Jaccard": {
            "mean": mean_jaccard,
            "class": jaccard_class
        },
        "HD95": {
            "mean": mean_hd95,
            "class": hd95_class
        },
        "ASD": {
            "mean": mean_asd,
            "class": asd_class
        }
    }

    raw_accumulators = {
        "schema": "utils_calc_metrics_fixed_raw_accumulators_v1",
        "dice_jaccard_aggregation": "global_pooled_voxel",
        "surface_aggregation": "mean_per_case_valid_cases",
        "class_ids": class_ids,
        "classes": [
            {
                "class_id": int(class_ids[idx]),
                "total_inter": float(total_inter[idx].item()),
                "total_pred": float(total_pred[idx].item()),
                "total_gt": float(total_gt[idx].item()),
                "dice": float(dice_class[idx]) / 100.0,
                "jaccard": float(jaccard_class[idx]) / 100.0,
                "hd95_sum": float(hd95_sum[idx].item()),
                "hd95_count": int(hd95_count[idx].item()),
                "asd_sum": float(asd_sum[idx].item()),
                "asd_count": int(asd_count[idx].item()),
            }
            for idx in range(num_eval_classes)
        ],
    }

    if debug_export_path is not None and _get_rank() == 0:
        debug_export_path = os.fspath(debug_export_path)
        debug_parent = os.path.dirname(os.path.abspath(debug_export_path))
        os.makedirs(debug_parent, exist_ok=True)
        with open(debug_export_path, "w", encoding="utf-8") as handle:
            json.dump(raw_accumulators, handle, indent=2)

    metric_name_map = {
        "dice": "Dice",
        "jaccard": "Jaccard",
        "iou": "Jaccard",
        "hd95": "HD95",
        "asd": "ASD"
    }

    selected_results = {}

    for m in metrics:
        if m in metric_name_map:
            selected_results[metric_name_map[m]] = results[metric_name_map[m]]

    if return_dict:
        if return_raw_accumulators:
            selected_results["_raw_accumulators"] = raw_accumulators
        return selected_results

    # 兼容原始调用方式
    if return_raw_accumulators:
        return mean_dice, dice_class, raw_accumulators
    return mean_dice, dice_class


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将归一化后的 Tensor 还原为 [0, 255] 的 uint8 numpy 数组 (H, W, C) RGB。

    如果你的 Dataset 用的是 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]，
    请修改默认参数。
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    arr = tensor.clone().numpy()

    # 常规 RGB: (3,H,W)
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = arr.transpose(1, 2, 0)
        img = img * np.array(std) + np.array(mean)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    # 灰度或非常规输入，尽量转成 2D uint8
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]

    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()

    return (arr * 255).astype(np.uint8)
