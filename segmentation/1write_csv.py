import os
import pandas as pd
from pathlib import Path
import os.path as osp
import argparse
from sklearn.model_selection import train_test_split
import collections
from utils.process_json import load_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='配置文件路径')
    args = parser.parse_args()
    return args


def check_data(data_root):
    if isinstance(data_root, str):
        data_root = [data_root]
    print("data_root: ", data_root)
    all_files = []
    for dr in data_root:
        all_files += [str(i) for i in Path(dr).glob('**/*.png')] + \
                     [str(i) for i in Path(dr).glob('**/*.jpg')]
    valid_files = []
    for im_path in all_files:
        if im_path.endswith('-mask.png'):
            continue
        mask_path_list = [im_path[:-4] + "-mask.png", im_path[:-4] + "-mask.jpg"]
        for mask_path in mask_path_list:
            if osp.exists(mask_path):
                valid_files.append([im_path, mask_path])

    return valid_files

def main(cfg):
    data_root = cfg['data_root']
    # save_path = cfg['split_csv_savepath']
    save_path = cfg['train_loader']['args']['data_dir']
    # if osp.exists(save_path):
    #     print("split_csv_path already exist, skip prepare data!")
    #     return
    ts_ratio = cfg['ts_ratio']
    all_files = check_data(data_root)
    train_files, ts_files = train_test_split(all_files, test_size=ts_ratio, random_state=42)
    ts_num_half = len(ts_files) // 2
    if cfg['split_num'] == 3:
        path_dict = {'train': train_files, 'val': ts_files[:ts_num_half],
                      'test': ts_files[ts_num_half:]}
    else:
        path_dict = {'train': train_files, 'val': ts_files}
    data_frame = collections.defaultdict(list)
    for k, v in path_dict.items():
        data_frame['path'] += [i[0] for i in v]
        data_frame['mask'] += [i[1] for i in v]
        data_frame['split'] += [k] * len(v)

    save_dir = osp.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    path_frame = pd.DataFrame(data_frame)
    path_frame.to_csv(save_path, index=False)
    print("Output data path: ", save_path)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_json(args.config)
    # cfg = {
    #     'data_root': [
    #         "/workspace/di_group1/atesi_sz/data_processed_special/segmentation/series_cut/EL",
    #         "/workspace/gitlab/liyu/data/ats_sq_refactor/cutter/el_line"
    #         ],
    #     'split_csv_savepath':'output/data/ats_sq_refactor/cutter/el_line/20221107.xlsx',
    #     'ts_ratio': 0.2,
    #     'split_num': 2
    # }
    print(cfg)
    main(cfg)