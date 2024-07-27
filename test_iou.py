from __future__ import print_function
import warnings
from datetime import datetime
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
from PIL import Image
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import FGVC
from datesets import get_trainAndtest
import os
from config import class_nums
from config import HyperParams
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cv2


def train():
    # output dir
    # Data
    trainset, testset = get_trainAndtest()
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    ####################################################
    print("dataset: ", HyperParams['kind'])
    print("backbone: ", HyperParams['arch'])
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[HyperParams['kind']])
    ####################################################

    net = FGVC(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    # checkpoint = torch.load("/home/yd123/project/LocalAttentionFdm/air_resnet50_output/best_model.pth")
    # checkpoint = torch.load(r"F:\graduateStudentTest\Paper2\New localization\bird\mean\best_model.pth")
    # checkpoint = torch.load(r"F:\graduateStudentTest\Paper2\New localization\air\resnet50\best_model.pth")
    # checkpoint = torch.load(r"/home/yd123/project/LocalAttentionFdm/car_resnet50_output/best_model.pth")
    checkpoint = torch.load(r"F:\graduateStudentTest\Paper2\New localization\car\resnet50\batchsize20_double_card\\best_model.pth")
    net.load_state_dict(checkpoint)
    net = net.cuda()
    netp = nn.DataParallel(net).cuda()
    val_acc = test(netp, testset, testloader)
    # print("current result: ", val_acc)


def imdenormalize(img, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]), to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def test(net, testset, testloader):
    net.eval()
    correct_com = 0
    total = 0
    targetsAll = []
    predictedAll = []
    softmax = nn.Softmax(dim=-1)
    res = 0
    cnt = 0

    is_bird = False
    for batch_idx, (inputs, targets, gt_box, file_path) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if gt_box == -1:
            continue
        image_path = os.path.join(testset.root, file_path[0])
        img = cv2.imread(image_path)
        H, W = img.shape[0], img.shape[1]
        w_ra, h_ra = 550.0 / W, 550.0 / H
        if is_bird:
            x, y, w, h = gt_box[0].item(), gt_box[1].item(), gt_box[2].item(), gt_box[3].item()
            new_x, new_y, new_w, new_h = w_ra * x, h_ra * y, w_ra * w, h_ra * h
            new_x = new_x - 51
            new_y = new_y - 51
            x0 = max(0, new_x)
            y0 = max(0, new_y)
            x1 = min(447, new_x + new_w)
            y1 = min(447, new_y + new_h)
            gt_bbox = (x0, y0, x1, y1)
        else:
            xx0, yy0, xx1, yy1 = gt_box
            x0, y0, x1, y1 = w_ra * xx0-51, h_ra * yy0-51, w_ra*xx1-51, h_ra *yy1-51
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(447, x1)
            y1 = min(447, y1)
            gt_bbox = (x0, y0, x1, y1)

        net.module.features.__setattr__('gt_bbox', gt_bbox)
        # origin_image = imdenormalize(inputs[0].cpu().numpy().transpose(1, 2, 0),
        #                              mean=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        #                              std=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        #                              to_bgr=True)
        # origin_image = np.uint8(origin_image * 255)
        #
        # cv2.rectangle(origin_image,(int(x0), int(y0)), (int(x1), int(y1)),(0,0,255),3)
        # cv2.imwrite('img_xxxxx.jpg',origin_image)
        # cv2.waitKey(0)

        iou = net(inputs)
        cnt = cnt + 1
        res = res + iou
    iou_out = res / cnt
    torch.set_printoptions(sci_mode=False, precision=6)
    print('%.6f' % cnt)
    print('%.6f' % res)
    print('%.6f' % iou_out)

    return iou_out


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


import os

if __name__ == '__main__':
    set_seed(666)
    # set_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = HyperParams['gpu']
    train()
