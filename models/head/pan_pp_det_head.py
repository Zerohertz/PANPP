import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..loss import build_loss, iou, ohem_batch
from ..post_processing import pa, boxgen
from ..utils import CoordConv2d


class PAN_PP_DetHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel,
                 loss_emb,
                 use_coordconv=False):
        super(PAN_PP_DetHead, self).__init__()
        if not use_coordconv:
            self.conv1 = nn.Conv2d(in_channels,
                                   hidden_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        else:
            self.conv1 = CoordConv2d(in_channels,
                                     hidden_dim,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        return out

    def get_results(self, out, img_meta, cfg):
        resize_const, pos_const, len_const = map(float, [2.0, 0.2, 0.5])
        results = {}
        if cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])

        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        label = pa(kernels, emb,
                   cfg.test_cfg.min_kernel_area / (cfg.test_cfg.scale**2))

        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        scale = np.array((float(org_img_size[1]) / float(img_size[1]), float(org_img_size[0]) / float(img_size[0])), dtype=np.float32)
        scale = scale*resize_const
        label = cv2.resize(label, (int(img_size[1]//resize_const), int(img_size[0]//resize_const)),
                           interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (int(img_size[1]//resize_const), int(img_size[0]//resize_const)),
                           interpolation=cv2.INTER_NEAREST)

        min_area = cfg.test_cfg.min_area / ((cfg.test_cfg.scale**2) * (resize_const**2))
        bboxes = boxgen(label, score, label_num, min_area, cfg.test_cfg.min_score, scale, pos_const, len_const)

        results['bboxes'] = bboxes
        return results

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances,
             gt_bboxes):
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        # loss_text = dice_loss(texts, gt_texts, selected_masks, reduce=False)
        loss_text = self.text_loss(texts,
                                   gt_texts,
                                   selected_masks,
                                   reduce=False)
        iou_text = iou((texts > 0).long(),
                       gt_texts,
                       training_masks,
                       reduce=False)
        losses = {'loss_text': loss_text, 'iou_text': iou_text}

        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i,
                                             gt_kernel_i,
                                             selected_masks,
                                             reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).long(),
                         gt_kernels[:, -1, :, :],
                         training_masks * gt_texts,
                         reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        loss_emb = self.emb_loss(embs,
                                 gt_instances,
                                 gt_kernels[:, -1, :, :],
                                 training_masks,
                                 gt_bboxes,
                                 reduce=False)
        losses.update(dict(loss_emb=loss_emb))

        return losses
