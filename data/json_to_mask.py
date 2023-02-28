from omegaconf import OmegaConf
from argparse import ArgumentParser, Namespace
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
        if label == '-1':
            return
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

    src_img = cv2.imread(img_path, 0)
    if osp.exists(json_path):
        with open(json_path, 'r') as f:
            ann = json.load(f)
        ann_shapes = ann["shapes"]
        if len(ann_shapes) > 0:
            mask = get_mask(src_img.shape, ann_shapes)
            if mask is None:
                # -1 or a, move 图片和json
                save_dir = os.path.join(ok_dir, 'a')
                os.makedirs(save_dir, exist_ok=True)
                shutil.move(img_path, save_dir)
                shutil.move(json_path, save_dir)
            else:
                # ng, 保存mask
                save_dir = ng_dir
                os.makedirs(save_dir, exist_ok=True)
                if save_dir == osp.dirname(img_path):
                    save_mask_name = im_name[:-4] + '-mask.jpg'
                else:
                    save_mask_name = im_name
                save_path = osp.join(save_dir, save_mask_name)
                mask = np.array(mask, dtype='uint8')
                cv2.imwrite(save_path, mask)
    else:
        # ok, move图片并保存mask
        save_dir = os.path.join(ok_dir, 'ok')
        os.makedirs(save_dir, exist_ok=True)
        shutil.move(img_path, save_dir)
        # save_mask_name = im_name + '-mask.jpg'
        # save_path = osp.join(save_dir, save_mask_name)
        # mask = np.zeros(src_img.shape, np.uint8)
        # cv2.imwrite(save_path, mask)


def walk_datapath_dir(data_root):
    im_paths_all = [str(i) for i in Path(data_root).glob('**/*.png')] + \
                   [str(i) for i in Path(data_root).glob('**/*.jpg')]
    # im_paths_all = [i for i in im_paths_all if osp.exists(i[:-4] + '.json')]
    assert im_paths_all, data_root
    return im_paths_all

def main(data_root, save_root, ng_dir):
    if ng_dir:
        save_ng_root = osp.join(save_root, ng_dir)
        data_root = osp.join(data_root, ng_dir)
    else:
        save_ng_root = save_root
    tmp = save_root if save_root[-1] != '/' else save_root[:-1]
    save_error_root = tmp + '_error'
    im_path_list = walk_datapath_dir(data_root)
    for im_path in im_path_list:
        # print(im_path)
        save_ng_dir = osp.dirname(im_path.replace(data_root, save_ng_root))
        process_one_img(im_path, save_ng_dir, save_error_root)


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a config")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = OmegaConf.load(args.config)
    data_root_list = config.data_root
    if isinstance(data_root_list, str):
        data_root_list = [data_root_list]
    for data_root in data_root_list:
        ng_dir = ''
        mask_dir = data_root

        # tmp = data_root if data_root[-1] != '/' else data_root[:-1]
        # mask_dir = tmp + '_mask'

        main(data_root, mask_dir, ng_dir)