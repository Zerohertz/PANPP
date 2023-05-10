import os
import os.path as osp
import zipfile
import cv2
import numpy as np
import datetime

class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = os.path.join(result_path, str(datetime.datetime.now()).replace('.','').replace(' ','-'))

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(self.result_path):
            os.makedirs(self.result_path)
            os.makedirs(self.result_path + '/score')
            os.makedirs(self.result_path + '/kernels')
            os.makedirs(self.result_path + '/emb')
            os.makedirs(self.result_path + '/label')

    def write_result(self, img_name, image_path, outputs):
        self._write_result_ic15(img_name,image_path, outputs)

    def _write_result_ic15(self, img_name, image_path, outputs):
        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']
        img=cv2.imread(image_path)
        file_name = '%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for i, bbox in enumerate(bboxes):
                poly = np.array(bbox).reshape((-1))
                poly[poly<0]=0
                strResult = '\t'.join([str(p) for p in poly])
                result=strResult + '\r\n'
                f.write(result)
                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        result_name='%s.jpg' % img_name
        cv2.imwrite(os.path.join(self.result_path, result_name), img)

        alpha = 0.5
        h, w = outputs['label'].shape
        img = cv2.imread(image_path)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        self._convert(img, outputs, 'score', result_name)
        self._convert(img, outputs, 'kernels', result_name)
        self._convert(img, outputs, 'emb', result_name)
        self._convert(img, outputs, 'label', result_name)

    def _convert(self, img, outputs, dir_name, result_name, alpha=0.5):
        tensor = outputs[dir_name]
        if len(tensor.shape) == 2:
            if dir_name == 'score':
                tensor -= tensor.min()
                tensor /= tensor.max()
                tensor *= 255
            elif dir_name == 'label':
                tensor[tensor > 0] = tensor[tensor > 0] / tensor[tensor > 0].max() * 254 + 1
            cmap = cv2.applyColorMap(tensor.astype(np.uint8), cv2.COLORMAP_JET)
            cmap[tensor == 0] = 0
            tensor = cv2.addWeighted(img, alpha, cmap, (1-alpha), 0)
            cv2.imwrite(os.path.join(self.result_path, dir_name, result_name), tensor)
        elif dir_name == 'kernels':
            h, w = outputs['label'].shape
            cmap = np.zeros((h,w,3), dtype=np.uint8)
            cmap[tensor[0,:,:]==1] = [255, 0, 0]
            cmap[tensor[1,:,:]==1] = [0, 0, 255]
            tensor = cv2.addWeighted(img, alpha, cmap, (1-alpha), 0)
            cv2.imwrite(os.path.join(self.result_path, dir_name, result_name), tensor)
        else:
            h, w = outputs['label'].shape
            palette = np.zeros((h*2,w*2,3), dtype=np.uint8)
            cnt = [[0,h,0,w], [0,h,w,w*2], [h,h*2,0,w], [h,h*2,w,w*2]]
            for i in range(tensor.shape[0]):
                tmp = tensor[i,:,:]
                tmp -= tmp.min()
                tmp /= tmp.max()
                tmp *= 255
                cmap = cv2.applyColorMap(tmp.astype(np.uint8), cv2.COLORMAP_JET)
                palette[cnt[i][0]:cnt[i][1],cnt[i][2]:cnt[i][3],:] = cv2.addWeighted(img, alpha, cmap, (1-alpha), 0)
            cv2.imwrite(os.path.join(self.result_path, dir_name, result_name), palette)