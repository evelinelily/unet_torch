# 转为voc格式的数据集
input_dir=
output_dir=/home/liyu/mnt/gitlab/liyu/data/
labels_file=labels.txt
python3 voc.py $input_dir $output_dir --labels $labels_file
