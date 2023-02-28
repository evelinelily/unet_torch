cfg_path=/workspace/gitlab/liyu/academy/stantard_pipeline/pytorch-segmentation_src/config/tongwei/frame_stitch.json
gpu_id=2


# python data/json_to_mask.py --config $cfg_path || \
# ! echo "生成mask失败！" || exit

cd segmentation/

# pip install -r simple_requirements.txt -i https://pypi.doubanio.com/simple
export PYTHONPATH=$PYTHONPATH:./


echo "################### begin training process ######################"
#
# echo '################ step 1: process data   #################'
python3 1write_csv.py --config $cfg_path
# #
# echo '############ step 2: begin training model ###############'
CUDA_VISIBLE_DEVICES=$gpu_id python3 2train.py --config $cfg_path

echo '############ step 3: export model to onnx, torchscript ###############'
python3 3export.py --config $cfg_path
