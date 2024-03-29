import torch.nn as nn
import torch
from torch import optim
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
import skimage.io as io

from model import lbdNet
from PointRend import PointHead
from metric import get_dc, get_dc1, layer_recoding
from util import decode_labels, fusion
from loss import CombinedLoss, ContrastiveLoss
from sampling_points import point_sample
import torch.optim.lr_scheduler as lr_scheduler

class Solver(nn.Module):
    def __init__(self, config, train_loader, test_loader):
        super(Solver, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.pretrained_model = config.pretrained_model
        self.num_epochs = config.num_epochs
        self.img_ch = config.img_ch
        self.num_classes = config.num_classes
        self.model_save_path = config.model_path
        self.epoch_start = 0
        self.best_test_dice = 0
        self.network_type = 'LBD_Net'
        self.save_path = config.data_path

        self.network = lbdNet(self.img_ch, self.num_classes)
        self.PointHead = PointHead()

        self.seg_criterion = CombinedLoss()
        self.ctr_criterion = ContrastiveLoss()
        self.optimizer = optim.Adam(self.network.get_grad_param(), lr=config.lr)
        self.network.to(self.device)
        self.PointHead.to(self.device)

    def train(self):
        if self.pretrained_model:
            self.epoch_start = int(self.pretrained_model.split('\\')[-1].split('.')[0].split('-')[1]) - 1
            self.network.load_state_dict(torch.load(self.pretrained_model))
            self.confi_network.load_state_dict(torch.load(self.pretrained_model_confi))
            print('loading pretrained model from %s' % self.pretrained_model)
        else:
            self.init_weights(self.network)

        for epoch in range(self.epoch_start, self.num_epochs):
            self.network.train()
            epoch_loss = 0
            num = len(self.train_loader)
            train_dice_list = []

            for i, data in enumerate(self.train_loader):
                images, gts, gts_dst, _, image_id = data
                images = images.to(self.device)
                gts = gts.long().to(self.device)
                gts_dst = gts_dst.to(self.device)

                self.optimizer.zero_grad()
                for param in self.network.parameters():
                    param.requires_grad = True

                fine, pred, dst_pred = self.network(images)

                result = self.PointHead(fine, pred)
                gt_points = point_sample(
                    gts.float().unsqueeze(1),
                    result["points"],
                    mode="nearest",
                    align_corners=False
                ).squeeze_(1).long()
                
                rpLoss = F.cross_entropy(result["rend"], gt_points, ignore_index=255)

                pred_s = F.softmax(pred.detach(), dim=1)
                dst_pred_t = torch.tanh(dst_pred)

                ctrLoss = self.ctr_criterion(fine, pred, gts)

                supLoss = self.seg_criterion(F.softmax(pred, dim=1), gts, dst_pred_t, gts_dst)

                loss = supLoss + 0.2 * ctrLoss + 0.8 * rpLoss
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                dice = get_dc(torch.argmax(pred_s.cpu(), 1), gts.cpu(), self.num_classes)
                train_dice_list.append(dice)

                print(
                    '[Epoch:%d/%dTrain Set:Layer] AvgDice:%.4f, [ILM~IPL:%.4f, INL~OPL:%.4f, ONL:%.4f, IS~OS:%.4f,RPE:%.4f,]' %
                    (epoch + 1, i, np.mean(dice), dice[0], dice[1], dice[2], dice[3], dice[4]))


            train_dice = list(np.mean(np.stack(train_dice_list), axis=0))
            avg_train_layer_dice = np.mean(train_dice)

            with torch.no_grad():
                self.network.eval()
                test_layer_dice_list = []
                for i, data in enumerate(self.test_loader):
                    images, layers_t, dst_t, _, img_id = data

                    images = images.to(self.device)

                    _, layers, dst = self.network(images)
                    layers = F.softmax(layers.cpu(), dim=1)

                    dice = get_dc(torch.argmax(layers, 1), layers_t, self.num_classes)
                    test_layer_dice_list.append(dice)

            test_dice = list(np.mean(np.stack(test_layer_dice_list), axis=0))
            avg_test_layer_dice = np.mean(test_dice)

            print(
                '[Train Set:Layer] AvgDice:%.4f, [ILM~IPL:%.4f, INL~OPL:%.4f, ONL:%.4f, IS~OS:%.4f,RPE:%.4f]' %
                (avg_train_layer_dice, train_dice[0], train_dice[1], train_dice[2], train_dice[3], train_dice[4]))

            print(
                '[Test  Set/layer] AvgDice:%.4f, [ILM~IPL:%.4f, INL~OPL:%.4f, ONL:%.4f, IS~OS:%.4f,RPE:%.4f]' %
                (avg_test_layer_dice, test_dice[0], test_dice[1], test_dice[2], test_dice[3], test_dice[4]))


            save_path = os.path.join(self.model_save_path, '{}-{}.pkl'.format(self.network_type, epoch + 1))
            torch.save(self.network.state_dict(), save_path)

    def test(self, layer_path, dst_path):
        assert self.pretrained_model
        if self.pretrained_model:
            self.network.load_state_dict(torch.load(self.pretrained_model))
            print('loading pretrained model from %s' % self.pretrained_model)

        with torch.no_grad():
            self.network.eval()
            test_dice_list = []
            for i, data in enumerate(self.test_loader):
                images, layers_t, dst_t, t_list, image_id = data
                image_id = image_id[0]

                images = images.to(self.device)
                fine, layers, dst = self.network(images)

                result = self.PointHead(fine, layers)
                layers = F.softmax(result["fine"], dim=1)
                fin_res = fusion(layers.cpu(), dst.cpu(), t_list)
                dice = get_dc(fin_res.cpu(), layers_t, self.num_classes)

                test_dice_list.append(dice)

                image = io.imread(os.path.join(self.save_path, 'test', 'images', image_id + '.png'))
                layers = fin_res.squeeze().numpy()
                layers = decode_labels(image, layers)
                layer_name = os.path.join(layer_path, image_id + '.png')
                io.imsave(layer_name, layers)

        test_dice = list(np.mean(np.stack(test_dice_list), axis=0))
        avg_dice = np.mean(test_dice)
        print(
            '[Test  Set/layer] AvgDice:%.4f, [ILM~IPL:%.4f, INL~OPL:%.4f, ONL:%.4f, IS~OS:%.4f,RPE:%.4f,]' %
            (avg_dice, test_dice[0], test_dice[1], test_dice[2], test_dice[3], test_dice[4]))

