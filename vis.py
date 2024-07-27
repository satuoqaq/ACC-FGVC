from __future__ import print_function
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import FGVC,SFEM
from datesets import get_trainAndtest

from config import class_nums
from config import HyperParams
import cv2
import os
from torchvision import transforms
from datesets import transform_test
from torchvision.datasets.folder import default_loader

def save_img(img_path, feature_map, name, num):
    if (len(feature_map.shape)):
        feature_map = feature_map[0]
    heatmap = feature_map.mean(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    o_img = cv2.imread(img_path)
    o_img = cv2.resize(o_img, (448, 448))
    gap = np.max(heatmap) - np.min(heatmap)
    heatmap = (heatmap - np.min(heatmap)) / gap
    heatmap = cv2.resize(heatmap, (448, 448))
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + o_img  # 这里的0.4是热力图强度因子
    file_name = os.path.join('vvv', str(num))
    if (os.path.exists(file_name) == 0):
        os.mkdir(file_name)
    cv2.imwrite(os.path.join(file_name, name + '.png'), superimposed_img)

def imdenormalize(img, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]), to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_feature_map(model, img_path, name, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    last_path = save_dir.split('/')[-1]
    ori_img = default_loader(img_path)
    o_img = cv2.imread(img_path)
    # cv2.imshow('x', o_img)
    # cv2.waitKey(0)
    model.draw_heatmap = True
    img = transform_test(ori_img).cuda()
    #
    origin_image = imdenormalize(img.cpu().numpy().transpose(1,2,0), mean=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                      std=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                      to_bgr=True)
    origin_image = np.uint8(origin_image*255)
    # l_path = r'F:\graduateStudentTest\Paper2\New localization\bird\vis\179.Tennessee_Warbler'
    # l_path = '/home/yd123/project/LocalAttentionFdm/143.Caspian_Tern'
    # if not os.path.exists(l_path):
    #     os.mkdir(l_path)
    # cv2.imwrite("/home/yd123/project/LocalAttentionFdm/122.Harris_Sparrow/origin_image.jpg", origin_image)
    cv2.imwrite(save_dir+r"/"+"origin_image.jpg", origin_image)
    c, h, w = img.shape
    img = img.reshape(1, c, h, w)
    featuremaps = model.features(img)
    i = 0
    for featuremap in featuremaps:
        # heatmap = featuremap_2_heatmap(featuremap)
        # heatmap = cv2.resize(heatmap, (ori_img.size[0], ori_img.size[1]))  # 将热力图的大小调整为与原始图像相同
        # heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
       # heatmap = cv2.resize(heatmap, (448, 448))
        # superimposed_img = heatmap * 0.4 + crop_448_img  # 这里的0.4是热力图强度因子
        # cv2.imwrite(os.path.join(save_dir, name + '_'+str(i) + '.png'), superimposed_img)  # 将图像保存到硬盘
        crop_448 = '/home/yd123/project/LocalAttentionFdm/012.Yellow_headed_Blackbird/' + last_path + '/crop_448.jpg'
        if i==0:
            conv_heat = model.conv_block1(featuremap)
            att1 = conv_heat
        elif i==1:
            conv_heat = model.conv_block2(featuremap)
            att2 = conv_heat
        else:
            conv_heat = model.conv_block3(featuremap)
            att3 = conv_heat
        save_img(crop_448, conv_heat, name + 'conv_block_' + str(i), save_dir)
        i = i + 1
        if i>=3:
            new_d1_from2, new_d2_from1 = model.inter(att1,att2)
            new_d1_from3, new_d3_from1 = model.inter(att1,att3)
            new_d2_from3, new_d3_from2 = model.inter(att2,att3)
            att1 = att1 + (new_d1_from2 + new_d1_from3)
            att2 = att2 + (new_d2_from1 + new_d2_from3)
            att3 = att3 + (new_d3_from1 + new_d3_from2)
            save_img(crop_448, att1, 'att1_heat_', save_dir)
            save_img(crop_448, att2, 'att2_heat_', save_dir)
            save_img(crop_448, att3, 'att3_heat_', save_dir)
            break



def main():
    # net = FGVC(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    net = FGVC(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    # checkpoint = torch.load("/home/yd123/project/LocalAttentionFdm/air_resnet50_output/best_model.pth")
    # checkpoint = torch.load("/home/yd123/project/LocalAttentionFdm/vis/best_model.pth")
    checkpoint = torch.load(r"F:\graduateStudentTest\Paper2\New localization\bird\mean\best_model.pth")
    net.load_state_dict(checkpoint)
    net = net.cuda()
    net.eval()
    # root_path = '/DATA1/data_mv/dataset/Fine-grained/stanford-cars/img/ 113'
    # root_path = '/DATA1/data_mv/dataset/Fine-grained/aircraft/train/Cessna 525'
    root_path = r'F:\graduateStudentTest\CUB_200_2011\images\067.Anna_Hummingbird'
    # F:\graduateStudentTest\CUB_200_2011\images\143.Caspian_Tern\Caspian_Tern_0018_146010.jpg
    # F:\graduateStudentTest\CUB_200_2011\images\072.Pomarine_Jaeger\Pomarine_Jaeger_0023_61431.jpg
    l_path = r'F:\graduateStudentTest\Paper2\New localization\bird\vis\Similar Vis\1\067Anna_Hummingbird_0086_56495'
    cnt = 1
    path = "Anna_Hummingbird_0086_56495.jpg"
    # for path in os.listdir(root_path):
    #     if not os.path.exists(l_path):
    #         os.mkdir(l_path)
    draw_feature_map(net, os.path.join(root_path, path), path, l_path + '/' + str(cnt))
    # draw_feature_map(net, os.path.join(root_path, path), path, l_path)
    #     cnt=cnt+1

    print('finished!')


if __name__ == '__main__':
    main()
    # os.environ['CUDA_VISIBLE_DEVICES'] = HyperParams['gpu']
