# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentation import get_training_augmentation
from skimage.io import  imread
from skimage.transform import resize

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        # class_name_file = os.path.join(self.root, 'class_names.txt')
        # class_names = [line.rstrip() for line in tuple(open(class_name_file, "r"))]
        # self.num_classes = len(class_names) - 1
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        label_path = os.path.join(self.label_dir, image_id + '.png')
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # label_path = os.path.join(self.label_dir, image_id + '.npy')
        # label = np.load(label_path)
        # print(label.shape)
        image = imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = resize(image, self.crop_size, mode='constant', preserve_range=True)
        label = resize(label, self.crop_size, mode='constant', preserve_range=True)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id

    # def _load_data(self, index):
    #     image_id = self.files[index]
    #     image_path = os.path.join(self.image_dir, image_id + '.jpg')
    #     image = np.asarray(Image.open(image_path), dtype=np.float32)
    #     # label_path = os.path.join(self.label_dir, image_id + '.png')
    #     # label = np.asarray(Image.open(label_path), dtype=np.int32)
    #     label_path = os.path.join(self.label_dir, image_id + '.npy')
    #     label = np.load(label_path)
    #     image_id = self.files[index].split("/")[-1].split(".")[0]
    #     return image, label, image_id

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        # if self.val:
        #     image, label = self._val_augmentation(image, label)
        if self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

class VOCAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, **kwargs):
        self.num_classes = 21
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))
    
    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            "training_augmentation": get_training_augmentation(crop_size)
        }
    
        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(VOC, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

