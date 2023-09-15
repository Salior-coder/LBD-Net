import numpy as np
import torch


def fusion(pred_prob, pred_dist, t_list):
    _, _, H, W = pred_prob.size()
    L_c = torch.argmax(pred_prob, dim=1)
    index = 0
    for i in range(6):
        BTDM_d = pred_dist[:, i]
        mask1 = torch.zeros_like(BTDM_d)
        mask2 = torch.zeros_like(BTDM_d)
        mask1[(BTDM_d > 2/t_list[i]) & (L_c > i)] = 1
        mask2[(BTDM_d < -(2/t_list[i+1])) & (L_c == i)] = 1
        L_c[mask1.bool()] = i
        L_c[mask2.bool()] = i + 1
        index += 1
    return L_c


def one_hot(label, num_classes):
    one_hot = np.zeros((num_classes, label.shape[0], label.shape[1]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[i, ...] = (label == i + 1)
    return one_hot


def batch_one_hot(labels, num_classes):
    list = []
    for i in range(labels.shape[0]):
        label = labels[i, ...].squeeze()
        one_hot = np.zeros((num_classes, label.shape[0], label.shape[1]), dtype=label.dtype)
        for i in range(num_classes):
            one_hot[i, ...] = (label == i)
        list.append(one_hot)
    one_hot = np.stack(list, axis=0)
    return one_hot


def decode_labels(image, label):
    """ store label data to colored image """

    layer1 = [255, 0, 0]
    layer2 = [255, 165, 0]
    layer3 = [255, 255, 0]
    layer4 = [0, 255, 0]
    layer5 = [0, 127, 255]
    layer6 = [0, 0, 255]
    layer7 = [127, 255, 212]
    layer8 = [139, 0, 255]

    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8])
    for l in range(0, 6):
        r[label == l] = label_colours[l, 0]
        g[label == l] = label_colours[l, 1]
        b[label == l] = label_colours[l, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.uint8(rgb)







