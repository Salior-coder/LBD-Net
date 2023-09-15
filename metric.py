import numpy as np
import torch


def layer_recoding(dst):
    dst[dst > 0] = 1
    dst[dst <= 0] = 0
    diff_tensor = torch.empty((1, 5, 400, 400), dtype=torch.float32)
    for i in range(5):
        diff_tensor[:, i, ...] = dst[:, i + 1, ...] - dst[:, i, ...]
        diff_tensor[:, i, ...] = torch.where(diff_tensor[:, i, ...] > 0, torch.tensor(i + 1), torch.tensor(0))
    return diff_tensor


def one_hot(label, num_classes):
    one_hot = np.zeros((num_classes, label.shape[0], label.shape[1]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[i, ...] = (label == i)
    return one_hot


# 边界路径计算dice或f1
def get_dc1(SR, GT, num_classes):
    SR = SR.numpy()
    GT = GT.numpy()
    batch_dice_score = []

    for i in range(SR.shape[0]):
        one_hot_SR = SR.squeeze(0)
        one_hot_GT = one_hot(GT[i, ...], num_classes)
        one_hot_GT = one_hot_GT[1:-1, ...]
        dice_socre = dice_cal(one_hot_GT, one_hot_SR, num_classes=num_classes - 2)
        batch_dice_score.append(dice_socre)
    #
    batch_dice_score = np.stack(batch_dice_score, axis=0)
    batch_mean_dice_score = np.mean(batch_dice_score, axis=0).squeeze()

    return batch_mean_dice_score


# 层路径计算dice或f1
def get_dc(SR, GT, num_classes):
    # SR(batch_size,h,w)
    SR = SR.numpy()
    GT = GT.numpy()

    batch_dice_score = []

    for i in range(SR.shape[0]):
        one_hot_SR = one_hot(SR[i, ...], num_classes)
        one_hot_GT = one_hot(GT[i, ...], num_classes)

        f1_socre = dice_cal(one_hot_GT, one_hot_SR, num_classes=num_classes)
        batch_dice_score.append(f1_socre)

    batch_dice_score = np.stack(batch_dice_score, axis=0)  # (4,2)

    batch_dice_score = np.delete(batch_dice_score, [0, num_classes - 1], axis=1)
    batch_mean_dice_score = np.mean(batch_dice_score, axis=0).squeeze()

    return batch_mean_dice_score


def get_error(pred, gt):
    return np.sum(np.abs(pred - gt))


def f1_score_metrix(one_hot_label, one_hot_pred, num_classes):
    epsilon = 1e-6
    one_hot_label = np.transpose(one_hot_label, (1, 2, 0))
    one_hot_pred = np.transpose(one_hot_pred, (1, 2, 0))
    flat_label = one_hot_label.reshape(-1, num_classes).astype(np.uint8).reshape(-1, num_classes)
    flat_pred = one_hot_pred.reshape(-1, num_classes).astype(np.uint8).reshape(-1, num_classes)
    TP = np.sum(flat_pred * flat_label, axis=0).astype(np.float)
    FP = np.sum(-flat_pred * (flat_label - 1), axis=0).astype(np.float)
    FN = np.sum(-(flat_pred - 1) * flat_label, axis=0).astype(np.float)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return f1_score


def dice_cal(one_hot_label, one_hot_pred, num_classes):
    epsilon = 1e-6
    one_hot_label = np.transpose(one_hot_label, (1, 2, 0))
    one_hot_pred = np.transpose(one_hot_pred, (1, 2, 0))
    flat_label = one_hot_label.reshape(-1, num_classes).astype(np.uint8).reshape(-1, num_classes)
    flat_pred = one_hot_pred.reshape(-1, num_classes).astype(np.uint8).reshape(-1, num_classes)
    inter = np.sum(flat_pred * flat_label, axis=0).astype(np.float)
    union = np.sum(flat_pred, axis=0).astype(np.float) + np.sum(flat_label, axis=0).astype(np.float)
    if union[1] == 0:
        dice = None
    else:
        dice = (2 * inter) / (union)
    return dice
