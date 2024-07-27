import torch
import torch.nn as nn
from config import HyperParams
import torch.nn.functional as F
import numpy as np
from skimage import measure
import cv2
import os
import torch
from skimage import measure


class AttenDrop(nn.Module):
    def __init__(self, drop_threshold=0.3, keep_prob=0.25):
        super(AttenDrop, self).__init__()
        if not (0 <= drop_threshold <= 1):
            raise ValueError("Drop threshold must be in range [0, 1].")
        self.drop_threshold = drop_threshold
        self.cnt = 0

    def _get_drop_mask(self, attention, drop_thr):
        b, c, w1, h1 = attention.shape
        attention_ = attention.view(b, -1)  # B*S
        max_val, _= torch.max(attention_, dim=1, keepdim=True)
        max_val = max_val.expand_as(attention_)
        max_val = max_val.view(b, c, w1, h1)
        thr_val = max_val * drop_thr
        val = (attention > thr_val).float()
        return val


    def forward(self, input_):
        attention_map = torch.mean(input_, dim=1, keepdim=True)
        drop_mask = self._get_drop_mask(attention_map, self.drop_threshold)

        return drop_mask



class OLM(nn.Module):
    def __init__(self):
        super(OLM, self).__init__()
        self.AttentiveDrop = AttenDrop()


    def forward(self, fms):
        M = self.AttentiveDrop(fms)


        coordinates = []
        for i, m in enumerate(M):
            mask_np = m.cpu().numpy().reshape(14, 14)
            component_labels = measure.label(mask_np)
            properties = measure.regionprops(component_labels)
            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))


            intersection = ((component_labels==(max_idx+1)).astype(int) )==1
            prop = measure.regionprops(intersection.astype(int))
            if len(prop) == 0:
                bbox = [0, 0, 14, 14]
                print('there is one img no intersection')
            else:
                bbox = prop[0].bbox


            x_lefttop = bbox[0] * 32-1   #(14*32=448)
            y_lefttop = bbox[1] * 32-1
            x_rightlow = bbox[2] * 32 - 1
            y_rightlow = bbox[3] * 32 - 1

            if x_lefttop < 0:
                x_lefttop = 0
            if y_lefttop < 0:
                y_lefttop = 0
            coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]

            coordinates.append(coordinate)
        return coordinates
