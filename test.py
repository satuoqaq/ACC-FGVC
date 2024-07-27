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
    checkpoint = torch.load("/home/yd123/project/LocalAttentionFdm/vis/best_model.pth")
    net.load_state_dict(checkpoint)
    net = net.cuda()
    netp = nn.DataParallel(net).cuda()
    val_acc = test(netp, testset, testloader)
    print("current result: ", val_acc)


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

    for batch_idx, (inputs, targets, gt_box, file_path) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        x, y, w, h = gt_box[0].item(), gt_box[1].item(), gt_box[2].item(), gt_box[3].item()
        image_path = os.path.join(testset.root, file_path[0])
        img = cv2.imread(image_path)
        H, W = img.shape[0], img.shape[1]
        w_ra, h_ra = 550.0 / W, 550.0 / H
        new_x, new_y, new_w, new_h = w_ra * x, h_ra * y, w_ra * w, h_ra * h
        new_x = new_x - 51
        new_y = new_y - 51

        x0 = max(0, new_x)
        y0 = max(0, new_y)
        x1 = min(447, new_x + new_w)
        y1 = min(447, new_y + new_h)

        gt_bbox = (x0, y0, x1, y1)
        net.module.features.__setattr__('gt_bbox',gt_bbox)
        # origin_image = imdenormalize(inputs[0].cpu().numpy().transpose(1, 2, 0),
        #                              mean=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        #                              std=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        #                              to_bgr=True)
        # origin_image = np.uint8(origin_image * 255)
        #
        # cv2.rectangle(origin_image,(int(x0), int(y0)), (int(x1), int(y1)),color=(0,0,255),thikcness=3)
        # cv2.imwrite('img_xxxxx.jpg',origin_image)
        # cv2.waitKey(0)

        with torch.no_grad():
            output_1, output_2, output_3, output_concat, outputs_local = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat + outputs_local
            # outputs_com = output_1 + output_2 + output_3

        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        # if predicted_com == targets.data:
        #     softmax_score = softmax(outputs_com)
        #     torch.set_printoptions(sci_mode=False, precision=6)
        #     top_scores, top_indices = torch.topk(softmax_score, 5, dim=1)
        #     l_path = "/home/yd123/project/LocalAttentionFdm/"
        #     img_name = testset.imgs["path"][batch_idx]
        #     with open(l_path + 'correct_top5_score.txt', 'a') as f:
        #         f.write("\n\n")
        #         f.write(str(img_name.split('/')[0]))
        #         f.write("\n")
        #         f.write(str(img_name.split('/')[1]))
        #         f.write("\n")
        #         f.write("Top5_score:  ")
        #     for i in top_scores[0]:
        #         torch.set_printoptions(sci_mode=False, precision=6)
        #         with open(l_path + 'correct_top5_score.txt', 'a') as f:
        #             f.write(str('%.6f' % i.item()))
        #             f.write("       ")
        #     with open(l_path + 'correct_top5_score.txt', 'a') as f:
        #         f.write("\n")
        #         f.write("Top5_class:  ")
        #     for j in top_indices[0]:
        #         with open(l_path + 'correct_top5_score.txt', 'a') as f:
        #             f.write("      ")
        #             f.write(str(j.item()))
        #             f.write("       ")
        #     with open(l_path + 'correct_top5_score.txt', 'a') as f:
        #         f.write("\n")
        #         f.write("target_class=" + str(targets.data.item()))
        #         f.write("      ")
        #         f.write("predict_calss=" + str(predicted_com.item()))
        # correct_com += predicted_com.eq(targets.data).cpu().sum()
        # targetsAll += targets.cpu()
        # predictedAll += predicted_com.cpu()
    test_acc_com = 100. * float(correct_com) / total

    # 将预测失败的图片保存进文件夹：
    # 计算预测失败的图片
    # fail_index = []
    # # targetsAll=5793,测试集有5794张图片，0-5793
    # for i in range(len(targetsAll)):
    #     if targetsAll[i] != predictedAll[i]:
    #         fail_index.append(i)
    #
    # # 将预测失败的图片保存进某指定文件夹内，后缀名为.jpg
    # # 注意测试集中每个类别的图像，比如第1张图像命名为1.jpg，最后一张图像命名为30.jpg，其坐标范围应该是从0-29！！！！
    # # 即20.jpg对应的是第19张图像！！！
    # l_path = "/home/yd123/project/LocalAttentionFdm//"
    # with open(l_path + 'fail.txt', 'w') as f:
    #     f.write("fail_imgs numbers:" + str(fail_index.__len__()))
    #     f.write("\n")
    #     f.write("test acc:" + str(test_acc_com))
    #     f.write("\n")
    # for i in fail_index:
    #     img_name = testset.imgs["path"][i]
    #     path ="/DATA1/data_mv/dataset/Fine-grained/CUB_200_2011/images//"
    #     img = Image.open(path+img_name)
    #     with open(l_path+'fail.txt','a') as f:
    #         f.write(img_name)
    #         f.write("       ")
    #         f.write("target_class="+str(targetsAll[i].item()))
    #         f.write("       ")
    #         f.write("predict_calss=" + str(predictedAll[i].item()))
    #         f.write("\n")
    #
    #     # 检查文件夹是否存在
    #     str0="fail_img/"
    #     str1=img_name.split('/')[0]     # 如'001.Black_footed_Albatross'
    #     str2=img_name.split('/')[1]     # 如'Black_Footed_Albatross_0049_796063.jpg'
    #     if not os.path.exists(l_path + str0 + str1):
    #         os.makedirs(l_path + str0 + str1)
    #     img.save(l_path + str0 + str1 + '/' + str2)
    #     # img.save(l_path + 'fail_img/fail_img_' + str(i) + '.jpg')

    return test_acc_com


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
