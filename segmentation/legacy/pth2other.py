import torch
import json
import models
from collections import OrderedDict
import os
# from torchsummary import summary


def save_pt(net, input, save_path):
    print(3)
    traced_script_module = torch.jit.trace(net, input)
    print(2)
    traced_script_module.save(save_path)

def save_onnx(model, dummy_input, save_path):
    torch.onnx._export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: 'batch_size'},
                      "output": {0: 'batch_size'}},
        opset_version=11,
    )



num_classes = 2
json_dir = "/disk1/tmp/zhangyuechao/1CODA/seg/pytorch-segmentation/config_unet_tiny_white.json"
model_dir = '/disk1/tmp/zhangyuechao/3detect_result/seg/WHITE/UNET/12/UNetTINY/06-29_01-39/checkpoint-epoch810.pth'


config = json.load(open(json_dir))
# Model
model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
availble_gpus = list(range(torch.cuda.device_count()))
device = torch.device("cpu")#('cuda:0' if len(availble_gpus) > 0 else 'cpu')
print(model)

# Load checkpoint
checkpoint = torch.load(model_dir, map_location="cpu")#device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']
# If during training, we used data parallel
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
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
dummy_input = torch.Tensor(1, 3, 224, 224).to(device) # bs,c,h,w # torch.Tensor(1, 3, 256, 256).cuda() #
save_pt(model, dummy_input, os.path.dirname(model_dir) + '/epoch1820.pt')
save_onnx(model, dummy_input, os.path.dirname(model_dir) + '/epoch1820.onnx')














