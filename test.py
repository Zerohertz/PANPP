import argparse
import json
import os
import os.path as osp
import sys

import torch
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import AverageMeter, Corrector, ResultFormat, Visualizer

import time
import csv


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para / 1e6))
    print('-' * 90)


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    print('Start testing %d images' % len(test_loader))
    cfg.debug = False
    cfg.report_speed = False
    
    for idx, data in enumerate(test_loader):
        print('Testing %d/%d\r' % (idx, len(test_loader)), end='', flush=True)

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(cfg=cfg))
        
        start = time.time()
        # forward
        with torch.no_grad():
            outputs = model(**data)
        end = time.time()
        with open('./results/time/tmp.csv', 'a', encoding='utf8') as f:
            wr = csv.writer(f)
            wr.writerow([end - start])

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        image_path=test_loader.dataset.img_paths[idx]
        rf.write_result(image_name, image_path, outputs)

    print('Done!')


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    print(json.dumps(cfg._cfg_dict, indent=4))

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=data_loader.voc,
                char2id=data_loader.char2id,
                id2char=data_loader.id2char,
            ))
    model = build_model(cfg.model)
    model = model.cuda()

    if cfg.test_cfg.pretrain is not None:
        if os.path.isfile(cfg.test_cfg.pretrain):
            print("Loading model and optimizer from checkpoint '{}'".format(
                cfg.test_cfg.pretrain))

            checkpoint = torch.load(cfg.test_cfg.pretrain)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model_structure(model)
    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', default='config/pan_pp/pan_pp_test.py', help='config file path')
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()
    main(args)
