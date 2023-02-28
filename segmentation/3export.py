import torch
import json
import models
from collections import OrderedDict
import os
from pathlib import Path
import argparse
from utils.process_json import load_json
# from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()
    return args


def save_pt(net, input, save_path):
    # print(3)
    traced_script_module = torch.jit.trace(net, input)
    # print(2)
    traced_script_module.save(save_path)


def save_onnx(model, dummy_input, save_path):
    torch.onnx._export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )


def get_latest_weight(save_dir):
    model_paths = [str(i) for i in Path(save_dir).glob("**/*.pth")]
    model_path = sorted(model_paths, key=lambda fn: os.path.getmtime(fn))[-1]
    return model_path


def main(num_classes, model_path, input_shape):
    model_path_clean = model_path.split(".")[0]
    h, w = input_shape
    # Model
    model = getattr(models, config["arch"]["type"])(
        num_classes, **config["arch"]["args"]
    )
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cpu")  # ('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    print(model)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")  # device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    # If during training, we used data parallel
    if "module" in list(checkpoint.keys())[0] and not isinstance(
        model, torch.nn.DataParallel
    ):
        # for gpu inference, use data parallel
        # if "cuda" in device.type:
        #     model = torch.nn.DataParallel(model)
        # else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict
    # print(checkpoint)
    # load
    model.load_state_dict(checkpoint)
    # torch.save(model, "dist.pt")
    # print(model)
    model.to(device)
    model.eval()
    # summary(model, (3, 256, 1024)) # h,w
    #
    dummy_input = torch.Tensor(1, 3, h, w).to(
        device
    )  # bs,c,h,w # torch.Tensor(1, 3, 256, 256).cuda() #
    save_pt(model, dummy_input, model_path_clean + ".pt")
    save_onnx(model, dummy_input, model_path_clean + ".onnx")

if __name__ == "__main__":
    args = parse_args()
    config = load_json(args.config)
    num_classes = 2
    model_path = get_latest_weight(config["trainer"]["save_dir"])
    input_shape = config["train_loader"]["args"]["crop_size"]
    main(num_classes, model_path, input_shape)
