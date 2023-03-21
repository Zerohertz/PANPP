import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from mmcv import Config

from models import build_model

import onnx
from torch2trt import torch2trt


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
    cfg = Config.fromfile('config/TensorRT_cfg.py')
    model = build_model(cfg.model)
    model = model.eval().cuda()
    
    checkpoint = torch.load(cfg.test_cfg.pretrain)
    d = dict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        d[tmp] = value
    model.load_state_dict(d)

    data = prepare_test_data(cfg.data)
    inputData = data['imgs'].cuda()
    metaData = data['img_metas']
    
    with torch.no_grad():
        outputData = model(inputData)
    print(outputData)
    
    bboxes = model.det_head.get_results(outputData, metaData)
    print(bboxes)

    dynamic_axes = {
        'Input Image': {
            0: 'batch',
            2: 'Width',
            3: 'Height'
        },
        'Text Region, Text Kernel, Instance Vectors': {
            0: 'batch',
            2: 'Height',
            3: 'Width'
        }
    }

    torch.onnx.export(
        model,
        inputData,
        "test.onnx",
        input_names=["Input Image"],
        output_names=["Text Region, Text Kernel, Instance Vectors"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )

    '''
    trtexec --onnx=test.onnx \
        --saveEngine=test.plan \
        --minShapes=input:1x3x1024x1024 \
        --optShapes=input:1x3x1536x1536 \
        --maxShapes=input:1x3x2048x2048 \
        --shapes=input:1x3x1536x1536
    '''

    path = "./test.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), "test_inf.onnx")

    min_shape = [(1, 3, 1024, 1024)]
    max_shape = [(1, 3, 2048, 2048)]
    opt_shape = [(1, 3, 1536, 1536)]

    trt_model = torch2trt(
        model,
        [inputData],
        input_names=["Input Image"],
        output_names=["Text Region, Text Kernel, Instance Vectors"],
        fp16_mode=False,
        use_onnx=True,
        min_shapes=min_shape,
        max_shapes=max_shape,
        opt_shapes=opt_shape,
        onnx_opset=11,
    )
    
    torch.save(trt_model.state_dict(), "test.pth")
    torch.save(trt_model.state_dict(), "test.trt")
    engine_path = "test.engine"
    with open(engine_path, "wb") as f:
        f.write(trt_model.engine.serialize())
