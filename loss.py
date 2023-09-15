import torch.nn as nn
import torch
import torch.nn.functional as F
from sampling_points import point_sample



def ctrLoss(anchor, positive, negative):
    B, _, Na = anchor.shape
    _, _, Np = positive.shape
    _, _, Nn = negative.shape

    anchors_flat = anchor.view(B, -1, Na)
    positives_flat = positive.view(B, -1, Np)
    negatives_flat = negative.view(B, -1, Nn)

    cos_sim_pos = F.cosine_similarity(anchors_flat.unsqueeze(3), positives_flat.unsqueeze(2), dim=1)
    pos_sum = torch.exp(cos_sim_pos).sum(dim=2)
    cos_sim_neg = F.cosine_similarity(anchors_flat.unsqueeze(3), negatives_flat.unsqueeze(2), dim=1)
    neg_sum = torch.exp(cos_sim_neg).sum(dim=2)
    loss_ctr = (-1 / Na) * torch.mean(torch.log(pos_sum / neg_sum).sum(dim=1))

    return loss_ctr


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, fine, coarse, GT):
        assert coarse.dim() == 4, "Dim must be N(Batch)CHW"
        device = coarse.device
        B, _, H, W = coarse.shape
        mask, _ = coarse.sort(1, descending=True)
        H_step, W_step = 1 / H, 1 / W
        loss_ctr = 0
        for i in range(GT.max() + 1):
            certainty_map = mask[:, 0] - mask[:, 1]
            _, anc = certainty_map[GT == i].view(B, -1).topk(128, dim=1)
            uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
            _, pos = uncertainty_map[GT == i].view(B, -1).topk(256, dim=1)
            _, neg = certainty_map[GT != i].view(B, -1).topk(6 * 256, dim=1)

            anc_point = torch.zeros(B, 128, 2, dtype=torch.float, device=device)
            anc_point[:, :, 0] = W_step / 2.0 + (anc % W).to(torch.float) * W_step
            anc_point[:, :, 1] = H_step / 2.0 + torch.div(anc, W, rounding_mode='trunc').to(torch.float) * H_step

            pos_point = torch.zeros(B, 256, 2, dtype=torch.float, device=device)
            pos_point[:, :, 0] = W_step / 2.0 + (pos % W).to(torch.float) * W_step
            pos_point[:, :, 1] = H_step / 2.0 + torch.div(pos, W, rounding_mode='trunc').to(torch.float) * H_step

            neg_point = torch.zeros(B, 6 * 256, 2, dtype=torch.float, device=device)
            neg_point[:, :, 0] = W_step / 2.0 + (neg % W).to(torch.float) * W_step
            neg_point[:, :, 1] = H_step / 2.0 + torch.div(neg, W, rounding_mode='trunc').to(torch.float) * H_step

            anc_feature = torch.cat(
                [point_sample(coarse, anc_point, align_corners=False),
                 point_sample(fine, anc_point, align_corners=False)],
                dim=1)
            pos_feature = torch.cat(
                [point_sample(coarse, pos_point, align_corners=False),
                 point_sample(fine, pos_point, align_corners=False)],
                dim=1)
            neg_feature = torch.cat(
                [point_sample(coarse, neg_point, align_corners=False),
                 point_sample(fine, neg_point, align_corners=False)],
                dim=1)
            loss_ctr += ctrLoss(anc_feature, pos_feature, neg_feature)
        loss_ctr /= GT.max() + 1
        return loss_ctr


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss()
        self.dst_loss = nn.MSELoss()

    def forward(self, pred_layer, layer, pred_dst, dst):
        ce_loss = self.ce_loss(pred_layer, layer)
        dice_loss = self.dice_loss(pred_layer, layer)
        dst_loss = self.dst_loss(pred_dst, dst)
        loss = 0.5 * ce_loss + 0.5 * dice_loss + dst_loss
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        batch_size = targets.size(0)
        smooth = 1
        inputs_flat = inputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        nume = torch.sum(inputs_flat * targets_flat, 1) + smooth
        deno = inputs_flat.sum(1) + targets_flat.sum(1) + smooth
        loss = 2 * nume / deno
        loss = 1 - loss.sum() / batch_size
        return loss


def one_hot(label, depth):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    shape = label.size()
    one_hot = torch.FloatTensor(shape[0], depth, shape[1], shape[2]).zero_().cuda()
    label = label.view(shape[0], 1, shape[1], shape[2])
    one_hot = one_hot.scatter_(1, label, 1)
    return one_hot


class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, inputs, targets, weights=None):
        if weights is not None:
            weights = weights.squeeze()
        num_class = inputs.size(1)
        targets = one_hot(targets, num_class)
        dice = DiceLoss()
        total_loss = 0
        for i in range(num_class):
            dice_loss = dice(inputs[:, i], targets[:, i])
            if weights is not None:
                dice_loss *= weights[i]
            total_loss += dice_loss
        return total_loss
