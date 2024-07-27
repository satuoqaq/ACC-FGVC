import os
from localization.OLM import *
from torch.nn import init
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torch.nn import functional as F
from Resnet import resnet50
import Resnet
from model import *


class CFA_Block(nn.Module):
    def __init__(self, channels, h, w, reduction=16):
        super(CFA_Block, self).__init__()
        self.channel = channels
        self.feature_size = 512
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))  # 在x轴进行平均池化操作，x轴即为水平方向w，进而使w的值变为1
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))  # 在y轴进行平均池化操作，y轴为垂直方向h，进而使h的值变为1
        self.conv1_1x1 = nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1,
                                   bias=False)  # 图中的r即为reduction，进而使其输出的特征图像的通道数变为原先的1/16
        self.conv2_1x1 = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1,
                                   bias=False)  # 图中的r即为reduction，进而使其输出的特征图像的通道数变为原先的1/16
        self.relu = nn.ReLU()  # relu激活函数
        # self.bn1 = nn.BatchNorm2d(channels // reduction)  # 二维的正则化操作
        # self.bn2 = nn.BatchNorm2d(channels)

        self.F_h = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1,
                             bias=False)  # 将垂直方向上的通道数通过卷积来将其复原
        self.F_w = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1,
                             bias=False)  # 将水平方向上的通道数通过卷积来将其复原

        self.sigmoid_h = nn.Sigmoid()  # 定义的sigmoid方法
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x_h = self.avg_pool_x(x)
        x_w = self.avg_pool_y(x)
        x_h_conv = self.relu(self.conv1_1x1(x_h))
        x_w_conv = self.relu(self.conv1_1x1(x_w))
        mm_h_w = torch.matmul(x_h_conv, x_w_conv)
        mm_h_w_conv = self.relu(self.conv2_1x1(mm_h_w))
        importance = self.sigmoid(mm_h_w_conv)

        s_h = self.sigmoid_h(self.F_h(x_h_conv.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_w_conv))
        s_h_w = s_h.expand_as(x) * s_w.expand_as(x)
        att1 = x * importance
        att2 = x * s_h_w
        att = x + att1 + att2

        return att


class ResNet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(ResNet, self).__init__()
        if arch == 'resnet50':
            self.model = list(resnet50(pretrained=True, progress=True).children())
            self.pretrained_model = Resnet.resnet50(pretrained=True, progress=True)
        elif arch == 'resnet101':
            self.model = list(resnet101(pretrained=True, progress=True).children())
            self.pretrained_model = Resnet.resnet101(pretrained=True, progress=True)
        self.layer0_2 = nn.Sequential(*self.model[:6])
        self.layer3 = nn.Sequential(*self.model[6:7])
        self.layer4 = nn.Sequential(*self.model[7:8])
        # self.layer5 = nn.Sequential(*self.model[8:10])

        self.head1 = CFA_Block(512, 56, 56)
        self.head2 = CFA_Block(1024, 28, 28)
        self.head3 = CFA_Block(2048, 14, 14)

        self.OLM = OLM()


    def forward(self, x):
        fm1, fm2, fm, raw_embedding = self.pretrained_model(x)
        batch_size, channel_size, h, w = fm.shape
        coordinates = torch.tensor(self.OLM(fm.detach()))
        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to('cuda')  # [N, 3, 448, 448]
        for i in range(batch_size):
            x0, y0, x1, y1 = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]

        fm2 = self.layer0_2(local_imgs.detach())
        p2 = self.head1(fm2)
        fm3 = self.layer3(p2)
        p3 = self.head2(fm3)
        fm4 = self.layer4(p3)
        p4 = self.head3(fm4)

        return p2, p3, p4, raw_embedding

    def get_params(self):
        new_layers = list(self.head1.parameters()) + \
                     list(self.head2.parameters()) + \
                     list(self.head3.parameters())
        new_layers_id = list(map(id, new_layers))
        old_layers = filter(lambda p: id(p) not in new_layers_id, self.parameters())
        # old_layers = list(self.pretrained_model.parameters())
        return new_layers, old_layers
