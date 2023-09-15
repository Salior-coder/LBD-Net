import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label
from sklearn import preprocessing
from util import one_hot



# 将contour转换为距离图
def to_dist_map(gt, T_list, num_classes):
    gt = one_hot(np.array(gt), num_classes)
    l_list = []
    for i in range(0, gt.shape[0]):
        l_list.append(DistanceCal(gt[i, ...], T_list[i], T_list[i + 1]))
    gt = np.stack(l_list, 0)
    return gt


# 计算距离图的阈值
def gt_2_Threshold(gt, num_classes):
    gt = one_hot(np.array(gt), num_classes)
    l_list = []
    for i in range(0, gt.shape[0]):
        l_list.append(Threshold(gt[i, ...]))
    l_list.insert(0, l_list[0])
    l_list.append(l_list[-1])
    return l_list

def Threshold(image):
    min_height = float('inf')
    for col in range(image.shape[1]):
        start_height = None
        for row in range(image.shape[0]):
            if image[row, col] == 1:
                if start_height is None:
                    start_height = row
            elif start_height is not None:
                height = row - start_height
                if height < min_height:
                    min_height = height
    return min_height


# 距离图的计算公式
def DistanceCal(bin_image, T1, T2):
    bin_image = label(bin_image)
    result = np.ones_like(bin_image, dtype="float")
    for i in range(bin_image.shape[1]):
        flag = False
        for j in range(bin_image.shape[0]):
            if bin_image[j, i] == 1:
                flag = True
            if flag:
                result[j, i] = -1
    tmp1 = np.zeros_like(result, dtype='float')
    tmp1[bin_image == 0] = 1
    dist1 = distance_transform_edt(tmp1)
    dist1[(dist1 > T1) & (result == 1)] = T1
    dist1[(dist1 > T2) & (result == -1)] = T2
    x = preprocessing.MaxAbsScaler().fit_transform(dist1)
    dist = x * result
    result = dist.astype(float)
    result = result.astype('float32')
    return result





