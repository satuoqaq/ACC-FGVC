import os

import math

home = os.path.expanduser("~")
root_dirs = {
    'bird': r'/DATA1/data_mv/dataset/Fine-grained/CUB_200_2011/',
    # 'car': '/DATA1/data_mv/dataset/Fine-grained/stanford-cars',
    # 'air': '/DATA1/data_mv/dataset/Fine-grained/aircraft',
    # 'dog': '/DATA1/data_mv/dataset/Fine-grained/Stanford Dogs Dataset'
}

class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120
}

HyperParams = {
    'kind': 'bird',
    # 'kind':'air',
    # 'kind': 'car',
    # 'kind': 'dog',
    'epoch': 300,
    # 'arch': 'resnet101',
    # 'arch': 'densenet161',
    'arch': 'resnet50',
    'bs':20,
    'gpu': '0,1',
    'power': 1.5,
    'top_k': 5
}
#
