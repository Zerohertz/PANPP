import argparse
import time
import os
import csv
from tqdm import tqdm

import torch
from torch2trt import TRTModule

from mmcv import Config
import cv2
import numpy as np

from models import build_model

from TensorRT import prepare_test_data


def model_test(modelOps, model, img_name, img_path):
    data = prepare_test_data(img_path + img_name)
    inputData = data['imgs'].cuda()
    metaData = data['img_metas']
    ######################### Inference Start #########################
    inference_start = time.time()
    if modelOps == "torch":
        with torch.no_grad():
            outputData = model(inputData)
    else:
        outputData = model(inputData)
    inference_stop = time.time()
    ####################################################################
    with open(modelOps + '.csv', 'a', encoding='utf8') as f:
        wr = csv.writer(f)
        wr.writerow([(inference_stop - inference_start) * 1000])
    ####################################################################
    bboxes = torch_model.det_head.get_results(outputData, metaData)
    img = cv2.imread(img_path + img_name)
    file_name = img_name[:-4] + '.txt'
    file_path = './outputs/' + modelOps + '/'
    with open(file_path + file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            poly = np.array(bbox).astype(np.int32).reshape((-1))
            poly[poly<0]=0
            strResult = '\t'.join([str(p) for p in poly])
            result=strResult + '\r\n'
            f.write(result)
            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    result_name = img_name[:-4] + '.jpg'
    cv2.imwrite(file_path + '/' + result_name, img)
    
def main(modelOps):
    orgDir = os.getcwd()
    img_path = '/home/jovyan/local/1_user/hgoh@agilesoda.ai/PANPP/TestData/image/'
    os.chdir(img_path)
    img_names = []
    for i in os.listdir():
        if ('.jpg' in i) or ('.png' in i) or ('.tif' in i):
            img_names.append(i)
    
    os.chdir(orgDir)
    if modelOps == "torch":
        model = build_model(cfg.model)
        model = model.eval().cuda()
        checkpoint = torch.load(cfg.test_cfg.pretrain)
        d = dict()
        for key, value in checkpoint['state_dict'].items():
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d)
    elif modelOps == "pth":
        model = TRTModule()
        model.load_state_dict(torch.load('test.pth'))
    elif modelOps == "trt":
        model = TRTModule()
        model.load_state_dict(torch.load('test.trt'))
    elif modelOps == "engine":
        model = TRTModule()
        model.load_state_dict(torch.load('test.engine'))
    
    for img_name in tqdm(img_names):
        model_test(modelOps, model, img_name, img_path)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Opts')
    parser.add_argument('--modelOps')
    args = parser.parse_args()

    cfg = Config.fromfile('config/TensorRT_cfg.py')
    torch_model = build_model(cfg.model)
    
    main(args.modelOps)
