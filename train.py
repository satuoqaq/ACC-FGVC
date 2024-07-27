from __future__ import print_function
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import cv2
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import FGVC
from datesets import get_trainAndtest

from config import class_nums
from config import HyperParams
from torch.utils.tensorboard import SummaryWriter


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, power=2.0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.power = power

    def forward(self, input, target):
        # 计算自定义的 softmax
        custom_softmax = self.custom_softmax(input)
        # 计算每个样本的 log-probability
        log_prob = torch.log(custom_softmax)
        # 使用负对数似然损失函数计算损失
        loss = torch.mean(torch.sum(-log_prob.gather(1, target.view(-1, 1)), dim=1))
        return loss

    def custom_softmax(self, input):
        # 自定义的 softmax 函数，使用自定义的底数x次方进行归一化
        exp_input = torch.pow(self.power, input)
        normalized_input = exp_input / torch.sum(exp_input, dim=-1, keepdim=True)
        return normalized_input


class CustomCrossEntropyLossTopK(nn.Module):
    def __init__(self, topK_num=5, power=1.01):
        super(CustomCrossEntropyLossTopK, self).__init__()
        self.topK_num = topK_num
        self.power = power

    def forward(self, input, target):
        # Cross-Entropy
        intput_softmax = input.softmax(dim=-1)
        log_prob = torch.log(intput_softmax)

        # Improved Cross-Entropy
        TopK_input, TopK_index = torch.topk(input, self.topK_num, dim=-1)
        TopK_input_softmax = self.custom_softmax(TopK_input)
        log_prob_TopK = torch.log(TopK_input_softmax)

        log_prob_copy = torch.clone(log_prob)
        log_prob_copy.scatter_(1, TopK_index, log_prob_TopK)

        # 使用负对数似然损失函数计算损失
        loss1 = torch.mean(torch.sum(-log_prob.gather(1, target.view(-1, 1)), dim=1))
        loss2 = torch.mean(torch.sum(-log_prob_copy.gather(1, target.view(-1, 1)), dim=1))
        return loss1 + loss2

    def custom_softmax(self, input):
        # 自定义的 softmax 函数，使用自定义的底数x次方进行归一化
        exp_input = torch.pow(self.power, input)
        normalized_input = exp_input / torch.sum(exp_input, dim=-1, keepdim=True)
        return normalized_input

def train():
    # output dir
    output_dir = HyperParams['kind'] + '_' + HyperParams['arch'] + 'Ablation_output'
    try:
        os.stat(output_dir)
    except:
        os.makedirs(output_dir)
    # log dir
    # work_dirs, log_dir = 'work_dirs', 'tf_logs'
    work_dirs, log_dir = 'edge_work_dirs', 'edge_tf_logs'
    if not os.path.exists(work_dirs):
        os.mkdir(work_dirs)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_name = datetime.now().isoformat()[:-7].replace(':', '')
    log_path_name = os.path.join(work_dirs, log_name + '.txt')
    log_save_name = os.path.join(log_dir, log_name)
    os.mkdir(log_save_name)
    writer = SummaryWriter(log_save_name)

    f_txt = open(log_path_name, mode="w")

    # Data
    trainset, testset = get_trainAndtest()
    trainloader = DataLoader(trainset, batch_size=HyperParams['bs'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=HyperParams['bs'], shuffle=False, num_workers=8, pin_memory=False)

    ####################################################
    print("dataset: ", HyperParams['kind'])
    print("backbone: ", HyperParams['arch'])
    f_txt.writelines("backbone: " + HyperParams['arch'])
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[HyperParams['kind']])
    ####################################################

    net = FGVC(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])

    net = net.cuda()
    netp = nn.DataParallel(net).cuda()

    CELoss = CustomCrossEntropyLossTopK(topK_num=HyperParams['top_k'], power=HyperParams['power'])
    # CELoss = nn.CrossEntropyLoss()

    ########################
    new_params, old_params = net.get_params()
    new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
    old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
    new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer,
                                                                                HyperParams['epoch'], 0)

    old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer,
                                                                                HyperParams['epoch'], 0)

    max_val_acc = 0
    iterations = 0
    lr = 0.002

    print(HyperParams)
    f_txt.writelines(str(HyperParams))

    for epoch in range(0, HyperParams['epoch']):
        print('\nEpoch: %d' % epoch)
        f_txt.writelines('Epoch: %d\n' % epoch)
        start_time = datetime.now()
        print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))


        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            iterations = iterations + 1
            idx = batch_idx

            inputs, targets = inputs, targets.cuda()
            output_1, output_2, output_3, output_concat, local_logits = netp(inputs)

            # adjust optimizer lr
            new_layers_optimizer_scheduler.step()
            old_layers_optimizer_scheduler.step()

            # overall update
            loss1 = CELoss(output_1, targets) * 2
            loss2 = CELoss(output_2, targets) * 2
            loss3 = CELoss(output_3, targets) * 2
            concat_loss = CELoss(output_concat, targets)
            local_loss = CELoss(local_logits, targets)

            new_layers_optimizer.zero_grad()
            old_layers_optimizer.zero_grad()
            loss = loss1 + loss2 + loss3 + concat_loss + local_loss
            loss.backward()
            new_layers_optimizer.step()
            old_layers_optimizer.step()

            #  training log
            _, predicted = torch.max((output_1 + output_2 + output_3 + output_concat + local_loss).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item() + local_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()
            train_loss5 += local_loss.item()
            if batch_idx % 50 == 0 and batch_idx > 1:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss_local: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss5 / (batch_idx + 1),
                        train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))
                f_txt.writelines(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |  Loss_local: %.5f |Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss5 / (batch_idx + 1),
                        train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

                writer.add_scalar('LR/new_layer_lr', new_layers_optimizer_scheduler.get_lr()[0], iterations)
                writer.add_scalar('LR/old_layer_lr', old_layers_optimizer_scheduler.get_lr()[0], iterations)

                writer.add_scalar('Loss/loss_sum', train_loss / (batch_idx + 1), iterations)
                writer.add_scalar('Loss/loss_1', train_loss1 / (batch_idx + 1), iterations)
                writer.add_scalar('Loss/loss_2', train_loss2 / (batch_idx + 1), iterations)
                writer.add_scalar('Loss/loss_3', train_loss3 / (batch_idx + 1), iterations)
                writer.add_scalar('Loss/loss_concat', train_loss4 / (batch_idx + 1), iterations)
                writer.add_scalar('Loss/local_loss', train_loss5 / (batch_idx + 1), iterations)
                f_txt.flush()
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        print("train acc: ", train_acc)
        f_txt.writelines("train acc: %.3f\n" % (train_acc))
        # eval
        val_acc = test(net, testloader)
        torch.save(net.state_dict(), './' + output_dir + '/current_model.pth')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(net.state_dict(), './' + output_dir + '/best_model.pth')
        print("best result: ", max_val_acc)
        print("current result: ", val_acc)

        writer.add_scalar('Acc/train_acc', train_acc, epoch)
        writer.add_scalar('Acc/max_val_acc', max_val_acc, epoch)
        writer.add_scalar('Acc/val_acc', val_acc, epoch)
        end_time = datetime.now()
        print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))
        f_txt.writelines("best result: %.3f\n" % (max_val_acc))
        f_txt.writelines("current result: %.3f\n" % (val_acc))
        f_txt.writelines("end time: " + end_time.strftime('%Y-%m-%d-%H:%M:%S') + "\n")
    f_txt.close()


def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0
    softmax = nn.Softmax(dim=-1)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            output_1, output_2, output_3, output_concat, outputs_local = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat + outputs_local
            # outputs_com = output_1 + output_2 + output_3

        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

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
    # torch.use_deterministic_algorithms(True)
    train()
