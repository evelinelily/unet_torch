#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme

input_dir = [
    "/home/liyu/mnt/di_group1/atesi_sz/data_processed_special/segmentation/module_stit/VI",
    "/home/liyu/mnt/tmp_75/liyu/cache/0_zhong_qing_stitch/ann_res/"
]
output_dir = "/home/liyu/mnt/gitlab/liyu/data/zhong_qing/stitch/module_vi"
# labels_file = "label.txt"
vis_flag = 1
class_names = ['_background_', '1', '255']
class_name_to_id = {'__ignore__': -1, '_background_': 0, '1': 1, '2': 2, '255': 2}


def walk_dir(input_dir):
    json_path_list = []
    for dir in input_dir:
        json_path_list += glob.glob(osp.join(dir, "**/*.json"))
        assert json_path_list, dir
    return json_path_list

def main():
    json_path_list = walk_dir(input_dir)
    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        # sys.exit(1)
    else:
        os.makedirs(output_dir)
        os.makedirs(osp.join(output_dir, "JPEGImages"))
        os.makedirs(osp.join(output_dir, "SegmentationClass"))
        # os.makedirs(osp.join(output_dir, "SegmentationClassPNG"))
        if vis_flag:
            os.makedirs(
                osp.join(output_dir, "SegmentationClassVisualization")
            )
        print("Creating dataset:", output_dir)


    # for i, line in enumerate(open(labels_file).readlines()):
    #     class_id = i - 1  # starts with -1
    #     class_name = line.strip()
    #     if class_name in class_name_map:
    #         new_name = class_name_map[class_name]
    #     else:
    #         new_name = class_name
    #     class_name_to_id[new_name] = class_id
    #     if class_id == -1:
    #         assert class_name == "__ignore__"
    #         continue
    #     elif class_id == 0:
    #         assert class_name == "_background_"
    #     class_names.append(class_name)
    # class_names = tuple(class_names)
    print("class_names:", class_names)
    print("class_name_to_id", class_name_to_id)
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in json_path_list:
        if "161423_A4." not in filename:continue
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            output_dir, "SegmentationClass", base + ".npy"
        )
        # out_png_file = osp.join(
        #     output_dir, "SegmentationClassPNG", base + ".png"
        # )
        if vis_flag:
            out_viz_file = osp.join(
                output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        # labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)

        if vis_flag:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)

def split():
    from sklearn.model_selection import train_test_split
    ts_ratio = 0.2
    split_num = 2
    tmp_dir = os.path.join(output_dir, "ImageSets/Segmentation")
    os.makedirs(tmp_dir, exist_ok=True)
    all_files = [i.split('.')[0]+'\n' for i in os.listdir(osp.join(output_dir, "JPEGImages"))]
    train_files, ts_files = train_test_split(all_files, test_size=ts_ratio, random_state=42)
    ts_num_half = len(ts_files) // 2
    if split_num == 3:
        path_dict = {'train': train_files, 'val': ts_files[:ts_num_half],
                     'test': ts_files[ts_num_half:]}
    else:
        path_dict = {'train': train_files, 'val': ts_files}

    for k, v in path_dict.items():
        file_list = os.path.join(output_dir, "ImageSets/Segmentation", k + ".txt")
        with open(file_list, 'w') as f:
            f.writelines(v)

if __name__ == "__main__":
    # main()
    split()