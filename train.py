import argparse
import json
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from utils import AverageMeter

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
EPS = 1e-6

'''
from mmcv import Config
cfg = Config.fromfile('config/pan_pp_test.py')
from models import build_model
model = build_model(cfg.model)
import torch
model = torch.nn.DataParallel(model).cuda()
'''

def train(train_loader, model, optimizer, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter(max_len=500)
    data_time = AverageMeter(max_len=500)

    losses = AverageMeter(max_len=500)
    losses_text = AverageMeter(max_len=500)
    losses_kernels = AverageMeter(max_len=500)
    losses_emb = AverageMeter(max_len=500)
    losses_rec = AverageMeter(max_len=500)

    ious_text = AverageMeter(max_len=500)
    ious_kernel = AverageMeter(max_len=500)
    accs_rec = AverageMeter(max_len=500)

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(cfg=cfg))

        # forward
        outputs = model(**data)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item(), data['imgs'].size(0))

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item(), data['imgs'].size(0))
        if 'loss_emb' in outputs.keys():
            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item(), data['imgs'].size(0))
            loss = loss_text + loss_kernels + loss_emb
        else:
            loss = loss_text + loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item(), data['imgs'].size(0))
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item(), data['imgs'].size(0))

        losses.update(loss.item(), data['imgs'].size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            length = len(train_loader)
            log = f'({iter + 1}/{length}) ' \
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} | ' \
                  f'Batch: {batch_time.avg:.3f}s | ' \
                  f'Total: {batch_time.avg * iter / 60.0:.0f}min | ' \
                  f'ETA: {batch_time.avg * (length - iter) / 60.0:.0f}min | ' \
                  f'Loss: {losses.avg:.3f} | ' \
                  f'{losses_text.avg:.3f}/{losses_kernels.avg:.3f}/' \
                  f'{losses_emb.avg:.3f}' \
                  f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f}'
            print(log, flush=True)
    return {'LR': optimizer.param_groups[0]["lr"],
            'Loss': losses.avg,
            'Loss_text': losses_text.avg,
            'Loss_kernel': losses_kernels.avg,
            'Loss_emb': losses_emb.avg,
            'IoU (text)': ious_text.avg,
            'IoU (kernel)': ious_kernel.avg}


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1.0 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and
             state['epoch'] > cfg.train_cfg.epoch - 100 and
             state['epoch'] % 10 == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)


def main(args):
    cfg = Config.fromfile(args.config)
    cfg.update(dict(debug=False))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        num_workers=16,
        drop_last=True,
        pin_memory=True)

    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=data_loader.voc,
                char2id=data_loader.char2id,
                id2char=data_loader.id2char,
            ))
    model = build_model(cfg.model)

    model = torch.nn.DataParallel(model).cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg.train_cfg.lr,
                                        momentum=0.99,
                                        weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=cfg.train_cfg.lr)

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(
            cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        nmd = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in nmd}
        model.load_state_dict(pretrained_dict, False)
        for n, p in model.named_parameters():
            print(n, p.requires_grad)
            '''
            if 'fpem3' in n or 'fpem4' in n:
                p.requires_grad = True
            elif 'det_head' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
            '''
            if 'backbone' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
            print(n, p.requires_grad)
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    writer = SummaryWriter()
    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))
        
        tmp = train(train_loader, model, optimizer, epoch, start_iter, cfg)
        for i, j in tmp.items():
            writer.add_scalar(i, j, epoch)

        state = dict(epoch=epoch + 1,
                     iter=0,
                     state_dict=model.state_dict(),
                     optimizer=optimizer.state_dict())
        save_checkpoint(state, checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()
    main(args)
