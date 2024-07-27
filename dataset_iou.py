import numpy as np
import os
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from torchvision.datasets.utils import list_dir
import scipy.io as sio
from os.path import join
import pandas as pd
import imageio
from PIL import Image

transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class AIRDateset(Dataset):
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True):
        self.train = train
        self.root = root
        self.class_type = 'variant'
        self.split = 'trainval' if self.train else 'test'
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader
        self.bbox = pd.read_csv(os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', "images_box.txt"),
                                sep=" ", header=None, names=['idx', 'x0', 'y0', 'x1', 'y1'])
        self.gt_dict = {}
        for index, row in self.bbox.iterrows():
            print(row['idx'], row['x0'], row['y0'], row['x1'], row['y1'])
            self.gt_dict[row['idx']] = (row['x0'], row['y0'], row['x1'], row['y1'])
        # 对于每一行，通过列名访问对应的元素
        print("-----")
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        idx = path.split('/')[-1][:-4]
        if self.gt_dict.__contains__(int(idx)):
            gt_box = self.gt_dict[int(idx)]
        else:
            gt_box = -1
        sample = self.loader(path)
        if self.train:
            sample = transform_train(sample)
        else:
            sample = transform_test(sample)
        return sample, target, gt_box, path

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names

        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images

class CARDataSet(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.loader = default_loader
        self.train = train

        loaded_mat = sio.loadmat(os.path.join(self.root, "cars_annos.mat"))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        self.bbox = pd.read_csv(os.path.join(self.root, "annotation.txt"),
                                sep="\t", header=None, names=['idx', 'x0', 'y0', 'x1', 'y1', 'class'])
        self.gt_dict = {}
        for index, row in self.bbox.iterrows():
            # print(row['idx'], row['x0'], row['y0'], row['x1'], row['y1'], row['class'])
            self.gt_dict[row['idx']] = (row['x0'], row['y0'], row['x1'], row['y1'])

        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        idx = path.split('\\')[-1]
        if self.gt_dict.__contains__(os.path.join('car_ims',idx)):
            gt_box = self.gt_dict[os.path.join('car_ims',idx)]
        else:
            gt_box = -1

        image = self.loader(path)
        if self.train:
            image = transform_train(image)
        else:
            image = transform_test(image)
        return image, target,gt_box,path

    def __len__(self):
        return len(self.samples)

# class CARDataSet(Dataset):
#     def __init__(self, root, train=True, data_len=None):
#         self.root = root
#         self.is_train = train
#         train_img_path = os.path.join(self.root, 'cars_train')
#         test_img_path = os.path.join(self.root, 'cars_test')
#         label_file = open(os.path.join(self.root, 'class.txt'))
#         train_label_file = open(os.path.join(self.root, 'train.txt'))
#         test_label_file = open(os.path.join(self.root, 'test.txt'))
#         train_img_label = []
#         test_img_label = []
#         for line in train_label_file:
#             train_img_label.append(
#                 [os.path.join(train_img_path, line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1]) - 1])
#         for line in test_label_file:
#             test_img_label.append(
#                 [os.path.join(test_img_path, line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1]) - 1])
#         self.train_img_label = train_img_label[:data_len]
#         self.test_img_label = test_img_label[:data_len]
#
#         self.bbox = pd.read_csv(os.path.join(self.root, "annotation.txt"),
#                                 sep="\t", header=None, names=['idx', 'x0', 'y0', 'x1', 'y1', 'class'])
#         self.gt_dict = {}
#         for index, row in self.bbox.iterrows():
#             # print(row['idx'], row['x0'], row['y0'], row['x1'], row['y1'], row['class'])
#             self.gt_dict[row['idx']] = (row['x0'], row['y0'], row['x1'], row['y1'], row['class'])
#
#     def __getitem__(self, index):
#         idx = self.label_file[index][0].split('/')[-1]
#         path = os.path.join(self.root, 'car_ims',idx)
#         if self.gt_dict.__contains__(os.path.join('car_ims',idx)):
#             gt_box = self.gt_dict[os.path.join('car_ims',idx)]
#         else:
#             gt_box = -1
#         if self.is_train:
#             img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             img = Image.fromarray(img, mode='RGB')
#             img = transform_train(img)
#
#         else:
#             img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             img = Image.fromarray(img, mode='RGB')
#             img = transform_test(img)
#
#         return img, target,gt_box,path
#
#     def __len__(self):
#         if self.is_train:
#             return len(self.train_img_label)
#         else:
#             return len(self.test_img_label)


class CUBDataSet(Dataset):

    def __init__(self, root, train=True):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        bbox = pd.read_csv(os.path.join(root, "bounding_boxes.txt"), sep=" ", header=None,
                           names=['idx', 'x', 'y', 'w', 'h'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split, bbox], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.train = train

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        gt_box = (item['x'], item['y'], item['w'], item['h'])
        img = default_loader(os.path.join(self.root, file_path))
        if self.train:
            img = transform_train(img)
            return img, target, gt_box, file_path
        else:
            img = transform_test(img)
            return img, target, gt_box, file_path

    def __len__(self):
        return len(self.imgs)

class DOGDateSet(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.loader = default_loader
        self.train = train

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.train:
            image = transform_train(image)
        else:
            image = transform_test(image)
        return image, target

    def load_split(self):
        if self.train:
            split = sio.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = sio.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts

from config import HyperParams, root_dirs
def get_trainAndtest():
    kind = HyperParams['kind']
    root_dir = root_dirs[kind]
    if kind == 'bird':
        return CUBDataSet(root=root_dir, train=True), CUBDataSet(root=root_dir, train=False)
    elif kind == 'car':
        return CARDataSet(root=root_dir, train=True), CARDataSet(root=root_dir, train=False)
    elif kind == 'air':
        return AIRDateset(root=root_dir, train=True), AIRDateset(root=root_dir, train=False)
    elif kind == 'dog':
        return DOGDateSet(root=root_dir, train=True), DOGDateSet(root=root_dir, train=False)
    else:
        print("unsupported dataset")
        exit(0)
