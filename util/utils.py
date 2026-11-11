import numpy as np
import logging
import os

from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        # test
        # intersect_test = torch.sum(score[ignore != 1] * target[ignore != 1], dim=[1,2])
        # print("score.shape: ", score.shape)                     # score.shape： torch.Size([12, 252, 252])
        # print("intersect.shape: ", intersect.shape)             # torch.Size([])
        # print("intersect_test.shape: ", intersect_test.shape)   # intersect_test.shape:  torch.Size([1, 252])
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceLoss_sample_weight(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_sample_weight, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss_BS(self, score, target, ignore=None):
        target = target.float()
        smooth = 1e-5

        # 处理 ignore 掩码：如果为 None，则不忽略任何像素
        if ignore is not None:
            # 保留未被忽略的像素
            mask = (ignore != 1).float()  # shape: [B, H, W]
            score = score * mask
            target = target * mask

        # 在每个样本上计算交集与平方和
        intersect = torch.sum(score * target, dim=[1, 2])  # [B]
        y_sum = torch.sum(target * target, dim=[1, 2])  # [B]
        z_sum = torch.sum(score * score, dim=[1, 2])  # [B]

        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # [B]
        dice_loss = 1 - dice  # [B]

        return dice_loss  # [B]

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None, sample_weight=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        # 转换为 tensor 以便广播计算
        weight = torch.FloatTensor(weight).to(inputs.device)

        assert inputs.size() == target.size(), 'predict & target shape do not match'

        # VERSION BS
        dice_per_class = []
        for i in range(0, self.n_classes):
            dice_loss_cls_i = self._dice_loss_BS(inputs[:, i], target[:, i], ignore)
            dice_per_class.append(dice_loss_cls_i.unsqueeze(1))  # [B, 1]

        dice_all = torch.cat(dice_per_class, dim=1)  # [B, n_classes]

        # --- 修改开始：加入类别权重计算 ---
        # 假设 weight 是 [n_classes]
        # dice_all * weight 会广播为 [B, n_classes] * [n_classes] -> [B, n_classes]
        weighted_loss = dice_all * weight
        loss = weighted_loss.sum(dim=1) / weight.sum()  # [B] (每个样本加权后的平均loss)
        # --- 修改结束 ---

        # 加入样本权重
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1)
            loss = (loss * sample_weight).sum() / sample_weight.sum()
        else:
            loss = loss.mean()

        return loss


class DiceLoss_sample_weight_v1(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_sample_weight_v1, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss_BS(self, score, target, ignore=None):
        target = target.float()
        smooth = 1e-5

        # 处理 ignore 掩码：如果为 None，则不忽略任何像素
        if ignore is not None:
            # 保留未被忽略的像素
            mask = (ignore != 1).float()  # shape: [B, H, W]
            score = score * mask
            target = target * mask

        # 在每个样本上计算交集与平方和
        intersect = torch.sum(score * target, dim=[1, 2])  # [B]
        y_sum = torch.sum(target * target, dim=[1, 2])  # [B]
        z_sum = torch.sum(score * score, dim=[1, 2])  # [B]

        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # [B]
        loss = 1 - dice  # [B]

        return loss  # [B]

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None, sample_weight=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0

        # VERSION BS
        dice_per_class = []
        for i in range(0, self.n_classes):
            per_sample_loss  = self._dice_loss_BS(inputs[:, i], target[:, i], ignore)
            dice_per_class.append(per_sample_loss.unsqueeze(1))  # [B, 1]
        # 拼成 [B, num_classes]
        dice_all = torch.cat(dice_per_class, dim=1)
        loss = dice_all.mean(dim=1)  # 每个样本的平均 Dice loss，shape [B]
        # 加入样本权重
        if sample_weight is not None:
            sample_weight = sample_weight.view(-1)
            loss = (loss * sample_weight).sum() / sample_weight.sum()
        else:
            loss = loss.mean()

        return loss

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
