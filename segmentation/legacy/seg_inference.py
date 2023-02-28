
import numpy as np
import cv2
import torch
# import onnxruntime

from time import time
import math
import torch.nn.functional as F


class Unet(object):
    def __init__(self,
                 model_path='',
                 gpu_id='0',
                 gpu_mem_ratio=0.4,
                 gpu_type='nvidia',
                 name='',
                 model_type="onnx"):
        '''初始化函数
        Args：
            model_path: 模型路径
            gpu_id: '0'
            gpu_mem_ratio: GPU显存占用率
            product_shape: (未使用）产品尺寸类型, eg:[6,80]
        '''
        self.gpu_id = gpu_id
        self.gpu_mem_ratio = gpu_mem_ratio
        self.gpu_type = gpu_type
        self.name = name
        self.model_path = model_path
        self.model_type = model_type

        # 图像的均值和方差：
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        if self.gpu_type == 'nvidia':
            if self.model_type == "pt":
                availble_gpus = list(range(torch.cuda.device_count()))
                if gpu_id == -1:
                    self.device = torch.device("cpu")
                else:
                    self.device = torch.device('cuda:' + str(gpu_id) if len(availble_gpus) > 0 else 'cpu')
                self.model = torch.jit.load(model_path,  map_location=self.device)


                if self.device == "gpu":
                    self.model.cuda()
            elif self.model_type == "onnx":
                self.model = onnxruntime.InferenceSession(model_path)

        elif self.gpu_type == 'atlas':
            pass
    def resize(self, img, input_shape):
        """处理单张图片，
        注：img维度[h,w,c]， pytorch,onnx 的输入为[c,h,w]"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_shape)
        return img

    def normalization_batch(self, imgs):
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)
        imgs = imgs / 255.
        imgs -= self.MEAN
        imgs /= self.STD
        return imgs

    def data_prepro(self, imgs_list, input_shape):
        """数据预处理"""

        imgs_list = [self.resize(i, input_shape) for i in imgs_list]
        imgs_array = self.normalization_batch(imgs_list)
        #if self.gpu_type == "nvidia":
        # pytorch 预测的维度顺序为chw
        imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
        return imgs_array

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, img_array, atlas_manager):
        """
        Args:
            img_array: (n, h, w, c)维的ndarray
        Returns:
            y: (n, h, w, 1)
        """

        if self.gpu_type == 'nvidia':
            if self.model_type == "pt":
                img_array = torch.tensor(img_array)
                img_array = img_array.to(self.device)
                output = self.model(img_array)
                output = self.sigmoid(output.cpu().detach().numpy()) # output包含两个维度，第0维为ok类

            elif self.model_type == "onnx":
                img_array = np.ascontiguousarray(img_array, dtype=np.float32)
                ort_inputs = {self.model.get_inputs()[0].name: img_array}
                output = self.model.run(None, ort_inputs) # 输出的结果是list
                output = [self.sigmoid(ot) for ot in output] # output包含两个维度，第0维为ok类
            y = [np.squeeze(res)[1] for res in output]

        else:
            n, c, h, w = img_array.shape
            y = []
            for i, img in enumerate(img_array):
                img_norm = img.astype(np.float32)
                buffer = np.frombuffer(img_norm.tobytes(), np.float32)
                res, _ = atlas_manager.inference(self.name, [buffer])
                prediction = np.reshape(res[0], (2,  h, w))  # (2, 256, 256)
                mask = self.sigmoid(prediction)[1]
                y.append(mask)
        return y

    def extract_contours(self, mask):
        contour_res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour_res) == 2:
            contours = contour_res[0]
        elif len(contour_res) == 3:
            contours = contour_res[1]
        return contours

    def batch_predice(self, imgs, atlas_manager=[], batch_size=16):
        res = []
        total_size = len(list(imgs))
        batch_num = math.ceil(total_size / batch_size)
        for i in range(batch_num):
            if i < (batch_num - 1):
                batch_image_list = imgs[i * batch_size:(i + 1) * batch_size]
            else:
                batch_image_list = imgs[i * batch_size:]

            pred = self.predict(batch_image_list, atlas_manager)
            res.extend(pred)
        return res

    def inference(self, img_list, input_shape=(128, 128)):
        # 数据预处理
        t1 = time()
        self.image_array_seg = self.data_prepro(img_list.copy(), input_shape)
        # print("chuanjianju 数据处理时间： ", time() - t1)
        masks = self.batch_predice(self.image_array_seg)
        return masks


if __name__ == '__main__':
    from glob import glob
    import os
    from tqdm import tqdm


    model_root = "epoch1020.pt"
    root = "test/"
    output = "test_res/"
    if not os.path.exists(output):
        os.makedirs(output)

    ST = Unet(model_path=model_root, gpu_id=-1, gpu_mem_ratio=0.4, gpu_type='nvidia', name='', model_type="pt")

    img_dir_list = glob(root + '/' + "*.jpg")
    for imd in tqdm(img_dir_list):
        name = os.path.basename(imd)
        img = cv2.imread(imd)
        h, w = np.shape(img)[:2]
        mask = ST.inference([img])[0]
        mask = np.array(np.squeeze(mask), dtype='float32')
        mask = cv2.resize(mask, (w, h))
        mask = np.where(mask > 0.4, 1, 0)
        # if np.sum(mask) > 500:
        img_new = np.concatenate([img[:, :, 1],  mask * 255], 1)  # mask * img[:, :, 1],
        cv2.imwrite(output + "/%s" % (name), img_new)





