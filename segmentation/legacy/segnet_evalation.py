#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：code -> cut_evalation
@IDE    ：PyCharm
@Author ：Yuechao Zhang
@Date   ：2020/9/28 下午5:38
@Desc   ： 分割结果评估
==================================================
"""

import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
import cv2
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from sklearn.metrics import f1_score
from keras.models import Model, load_model
import tensorflow as tf

def mean_iou(input, target, classes = 1):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou/classes

def gt_er(gt):
    """
    对gt二值化
    :param gt: 真实标注
    :return:
    """
    mask = np.where(gt > 128, 1, 0)
    return mask
class evaluation(object):
    def __init__(self,model_path='',gpu_id=-1,gpu_mem_ratio=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_ratio
        self.sess = tf.Session(config=tf_config)
        self.graph = tf.get_default_graph()
        self.model = None
        self.model = load_model(model_path, )


    def predict(self, img, reize_size =(384, 128),th=0.5):
        size = np.shape(img)
        re_img = cv2.resize(img, reize_size)
        re_img = np.array(re_img, dtype='uint8')
        X_test = np.expand_dims(re_img, axis=0)
        preds_test = self.model.predict(X_test, verbose=1)
        preds_test = np.array(np.squeeze(preds_test), dtype='float32')
        preds_test = cv2.resize(preds_test, (size[1], size[0]))
        preds_test_upsampled_er = np.where(preds_test > th, 1, 0)
        return preds_test_upsampled_er

if __name__ == '__main__':
    all_iou = []
    model_path = '/home/zhangyuechao/nfs75_disk3/project_data/model_data/ats_series/seg/model/pintu_el/ats_chuanjian_pintu_el_v1.h5'
    big_dir = '/home/zhangyuechao/nfs75_disk3/project_data/model_data/ats_series/seg/el_pintu/test/'
    src_list = [i for i in glob.glob(big_dir + '*.jpg') if not i.endswith('mask.jpg')]
    EVA = evaluation(model_path=model_path)
    for img_path in tqdm(src_list):
        imgname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        pre_mask = EVA.predict(img)
        pre_mask = pre_mask.reshape(1, -1)
        gt = cv2.imread(img_path[:-4]+'-mask.jpg', 0)
        gt = gt_er(gt)
        gt = gt.reshape(1,-1)
        iou = mean_iou(pre_mask, gt, classes=1)
        all_iou.append(iou)
    #print(all_iou)
    print(np.mean(all_iou))
















































