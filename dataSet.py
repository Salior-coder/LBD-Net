import os
import torch
from torch.utils import data
from torchvision import transforms as T
from preprocess import to_dist_map, gt_2_Threshold
from skimage import io



class ImageFolder(data.Dataset):
    def __init__(self, root, mode='train'):

        self.root = root
        self.mode = mode

        self.train_path = os.path.join(root, 'train/images')
        self.test_path = os.path.join(root, 'test/images')
        self.gt_path = os.path.join(self.root, self.mode, 'labels')
        self.contour_path = os.path.join(self.root, self.mode, 'contours')

        self.train_list = [os.path.join(self.train_path, i) for i in os.listdir(self.train_path)]
        self.test_list = [os.path.join(self.test_path, i) for i in os.listdir(self.test_path)]


    def __getitem__(self, item):
        if self.mode == 'train':
            image_path = self.train_list[item]
        else:
            image_path = self.test_list[item]

        image_id = image_path.split('\\')[-1].split('.')[0]
        gt_image_path = os.path.join(self.gt_path, image_id + '.png')
        contour_image_path = os.path.join(self.contour_path, image_id + '.png')

        image = io.imread(image_path)
        gt = io.imread(gt_image_path)

        T_list = gt_2_Threshold(gt, gt.max() - 1)
        contour = io.imread(contour_image_path)
        gt_dst = to_dist_map(contour, T_list, 6)

        gt = torch.FloatTensor(gt)
        gt_dst = torch.FloatTensor(gt_dst)

        trans = T.Compose([T.ToTensor()])
        image = trans(image)

        return image, gt, gt_dst, T_list, image_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)


def get_loader(root, mode='trian', batch_size=4, num_workers=0):
    dataset = ImageFolder(root, mode)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return loader



