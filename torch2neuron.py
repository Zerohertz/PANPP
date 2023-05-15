import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from mmcv import Config

from models import build_model

import torch.neuron


def prepare_test_data(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print('Cannot read image: %s.' % img_path)
        raise
        
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )
    
    short_size = 1024
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    
    img_meta.update(dict(
        img_size=np.array(img.shape[:2])
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    data = dict(
        imgs=img.unsqueeze(0),
        img_metas=img_meta
    )
    return data

if __name__ == "__main__":
    cfg = Config.fromfile('config/Neuron_cfg.py')
    model = build_model(cfg.model)
    model = model.eval()
    
    checkpoint = torch.load(cfg.test_cfg.pretrain)
    d = dict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        d[tmp] = value
    model.load_state_dict(d)

    data = prepare_test_data(cfg.data)
    input_tensor = data['imgs']
    print(input_tensor.shape)
    meta_data = data['img_metas']

    with torch.no_grad():
        output_cpu = model(input_tensor)

    bboxes_cpu = model.det_head.get_results(output_cpu, meta_data)

    convert_neuron = False
    if convert_neuron:
        model_neuron = torch.neuron.trace(model, [input_tensor])
        filename = 'model_neuron.pt'
        # torch.jit.save(model_neuron, filename)
        model_neuron.save(filename)

    validate_neuron = False
    if validate_neuron:
        model_neuron = torch.jit.load('model_neuron.pt')
        output_neuron = model_neuron(input_tensor)
        bboxes_neuron = model.det_head.get_results(output_neuron, meta_data)
        print('Results of CPU (Tensors): ', output_cpu)
        print("Results of Neuron (Tensors): ", output_neuron)
        print('Results of CPU (bboxes): ', bboxes_cpu)
        print("Results of Neuron (bboxes): ", bboxes_neuron)