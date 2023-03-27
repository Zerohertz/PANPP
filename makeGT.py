from mmcv import Config
from utils import ResultFormat
import torch
from dataset import build_data_loader
import os.path as osp
import numpy as np


cfg = Config.fromfile('./config/pan_pp_test.py')
rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)
data_loader = build_data_loader(cfg.data.test)
test_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

out = {}
for idx, data in enumerate(test_loader):
    image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
    image_path = test_loader.dataset.img_paths[idx]
    try:
        out['bboxes'] = np.loadtxt(image_path.replace('image', 'txt').replace('jpg', 'txt').replace('png', 'txt').replace('tif', 'txt'),
                  delimiter='\t', usecols=[0,1,2,3,4,5,6,7])
    except Exception as e:
        print(e)
    try:
        len(out['bboxes'][0])
    except:
        out['bboxes'] = [out['bboxes']]
    rf.write_result(image_name, image_path, out)
