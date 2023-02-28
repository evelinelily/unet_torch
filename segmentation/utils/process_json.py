import json
import os
import os.path as osp

def merge_config(src, dst):
    # print(src)
    if isinstance(src, dict):
        for k, v in src.items():
            if k == "label_map":
                dst[k] = v
            elif k in dst:
                dst[k] = merge_config(src[k], dst[k])
            else:
                dst[k] = v
    elif isinstance(src, list):
        if isinstance(src[0], dict):
            dst[0] = merge_config(src[0], dst[0])
        else:
            dst = src
    else:
        dst = src
    return dst


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if 'parent' in cfg:
        project_path = osp.abspath('./')
        parent_path = osp.join(project_path, cfg['parent'])
        if osp.exists(parent_path):
            basic_cfg = load_json(parent_path)
            cfg = merge_config(cfg, basic_cfg)
        else:
            print(f"parent cfg path not exist: {project_path} and {cfg['parent']}")
    return cfg

def dump_json(save_path, all_params):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_params, f, indent=2, ensure_ascii=False)
