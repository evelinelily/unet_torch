# from dataprepro.utils import *
from glob import glob
from skimage.io import imsave
import shutil
from shape import *
import os
import cv2
import json
import numpy as np
import os.path as osp
from pathlib import Path


def get_mask(img_shape, shapes):
    mask_per_img = np.zeros(img_shape[:2], dtype=np.uint8)
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        mask_per_inst = shape_to_mask(img_shape[:2], points, shape_type)
        mask_per_img[mask_per_inst] = 255
    #     if label == '1':
    #         mask_per_img[mask_per_inst] = 255
    #     else:
    #         mask_0.append(mask_per_inst)
    # for mask_per_inst in mask_0:
    #         mask_per_img[mask_per_inst] = 0
    #     # cls[mask] = cls_id
    #     # ins[mask] = ins_id
    return mask_per_img


def process_one_img(img_path, ng_dir, ok_dir):
    json_path = img_path[:-4] + '.json'
    im_name = osp.basename(img_path)
    save_mask_name = im_name[:-4] + '-mask.jpg'
    src_img = cv2.imread(img_path, 0)
    mask = None
    if osp.exists(json_path):
        with open(json_path, 'r') as f:
            ann = json.load(f)
        ann_shapes = ann["shapes"]
        if len(ann_shapes) > 0:
            mask = get_mask(src_img.shape, ann_shapes)
            mask = np.array(mask,dtype='uint8')
            os.makedirs(ng_dir, exist_ok=True)
            save_path = osp.join(ng_dir, save_mask_name)
            cv2.imwrite(save_path, mask)
    if mask is None:
        print("okokokok", img_path)
        os.makedirs(ok_dir, exist_ok=True)
        shutil.move(img_path, ok_dir)
        mask = np.zeros(src_img.shape, np.uint8)
        save_path = osp.join(ok_dir, save_mask_name)
        cv2.imwrite(save_path, mask)

def walk_datapath_dir(data_root):
    im_paths_all = [str(i) for i in Path(data_root).glob('**/*.png')] + \
                   [str(i) for i in Path(data_root).glob('**/*.jpg')]
    im_paths_all = [i for i in im_paths_all if osp.exists(i[:-4] + '.json')]
    assert im_paths_all, data_root
    return im_paths_all

def main(data_dir_list, save_root):
    ng_root = osp.join(save_root, 'ng')
    ok_root = osp.join(save_root, 'ok')
    if isinstance(data_dir_list, str):
        data_dir_list = [data_dir_list]
    for data_root in data_dir_list:
        im_path_list = walk_datapath_dir(data_root)
        for im_path in im_path_list:
            # print(im_path)
            ng_path = im_path.replace(data_root, ng_root)
            ng_dir = os.path.dirname(ng_path)
            ok_path = im_path.replace(data_root, ok_root)
            ok_dir = os.path.dirname(ok_path)
            process_one_img(im_path, ng_dir, ok_dir)

if __name__ == '__main__':
    data_dir_list = '/home/liyu/mnt/gitlab/liyu/data/zhong_qing/stitch/module_vi/latest'
    save_root = '/home/liyu/mnt/gitlab/liyu/data/zhong_qing/stitch/module_vi/latest_mask'
    main(data_dir_list, save_root)
