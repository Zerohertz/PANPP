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

    def write_result(self, img_name,image_path, outputs):
        self._write_result_ic15(img_name,image_path, outputs)

    def _write_result_ic15(self, img_name,image_path, outputs):
        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']
#         scores = outputs['scores']
        words = None
        if 'words' in outputs:
            words = outputs['words']
        print()
        img=cv2.imread(image_path)
        file_name = '%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for i, bbox in enumerate(bboxes):        
                poly = np.array(bbox).astype(np.int32).reshape((-1))
                poly[poly<0]=0
                strResult = '\t'.join([str(p) for p in poly]) #+ ',' + str(scores[i]) 
                result=strResult + '\r\n'
                f.write(result)
                #f.write(str(scores[i]))
                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        result_name='%s.jpg' % img_name
        cv2.imwrite(os.path.join(self.result_path,result_name), img)

