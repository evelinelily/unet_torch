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
from glob import glob
from skimage.io import  imread
from skimage.transform import resize
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
vis = False


def round_clip_0_1(x,**kwargs):
    return x.round().clip(0, 1)

def brightness_adjustment(x,**kwargs):

    x_max = np.max(x)
    x_min = np.min(x)

    r = (np.random.random(1)-0.5)/5
    x = x*(1+r)
    x = np.clip(x, x_min, x_max)
    return x


# define heavy augmentations
def get_training_augmentation(imgsize):
    IMG_HEIGHT, IMG_WIDTH = imgsize
    train_transform = [
        A.HorizontalFlip(p=0.5),  #水平翻转
        A.VerticalFlip(p=0.5),
        A.IAAFliplr(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True, p=0.1),
        A.PadIfNeeded(IMG_HEIGHT, IMG_WIDTH),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Lambda(brightness_adjustment),
            ],
            p=0.5,),

        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


class VOCDataset(Dataset):
    def __init__(self,
                 root,
                 split,
                 mean,
                 std,
                 target_size=321,
                 return_id=False,
                 training_augmentation=None):

        self.num_classes = 2  # 其中背景为一类
        self.palette = palette.get_voc_palette(self.num_classes)
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.target_size = target_size
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id
        self.training_augmentation = training_augmentation

        cv2.setNumThreads(0)

    def _set_files(self):
        data = pd.read_csv(self.root)
        data_sp = data[data['split'] == self.split]
        self.mask_files = data_sp['mask'].values
        self.files = data_sp['path'].values
        if "train" in self.split:
            rand_idx = list(range(len(self.files)))
            np.random.shuffle(rand_idx)
            self.files = [self.files[i] for i in rand_idx]
            self.mask_files = [self.mask_files[i] for i in rand_idx]
        print(self.mask_files[:3], self.files[:3])
        print("load {} 数量: {}".format(self.split, len(self.files)))

    def _load_data(self, index):
        image_path = self.files[index]
        label_path = self.mask_files[index]
        image = imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = resize(image, self.target_size, mode='constant', preserve_range=True)
        mask = cv2.imread(label_path, 0)
        mask = resize(mask, self.target_size, mode='constant', preserve_range=True)
        mask = np.where(mask > 128, 1, 0)
        mask = np.array(mask)
        image_id = image_path.split("/")[-1].split(".")[0]
        return image, mask, image_id

    def _augmentation(self, image, mask):
        image = np.array(image, dtype='uint8')
        mask = np.array(mask, dtype='uint8')
        if "train" in self.split:
            sample = self.training_augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        image, label = self._augmentation(image, label)

        if vis:
            plt.subplot(121)
            plt.imshow(np.array(image,dtype='uint8'))
            plt.subplot(122)
            plt.imshow(label)
            plt.show()
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

class VOCself(BaseDataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 split,
                 crop_size=None,
                 num_workers=1,
                 val=False,
                 shuffle=False,
                 val_split=None,
                 return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'target_size': crop_size,
            'return_id': return_id,
            "training_augmentation": get_training_augmentation(crop_size)
        }

        if split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(VOCself, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


if __name__ == '__main__':
    import json
    import dataloaders
    config = json.load(open("/home/zhangyuechao/disk1/tmp/zhangyuechao/1CODA/seg/pytorch-segmentation/config.json"))
    train_loader = get_instance(dataloaders, 'val_loader', config)
    for next_input, next_target in train_loader:
        print(next_input, next_target)






