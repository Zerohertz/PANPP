import time

import torch
import torch.nn as nn

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class PAN_PP(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(PAN_PP, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)
        if neck.fpems == 3:
            self.fpem3 = build_neck(neck)
        elif neck.fpems == 4:
            self.fpem3 = build_neck(neck)
            self.fpem4 = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.rec_head = None
        if recognition_head:
            self.rec_head = build_head(recognition_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        upsample = nn.Upsample(size=(H // scale, W // scale), mode='bilinear')
        return upsample(x)

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        # backbone
        f = self.backbone(imgs)

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)
        try:
            f1, f2, f3, f4 = self.fpem3(f1, f2, f3, f4)
        except:
            pass
        try:
            f1, f2, f3, f4 = self.fpem4(f1, f2, f3, f4)
        except:
            pass

        # FFM
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        # detection
        out_det = self.det_head(f)

        if self.training:
            out_det = self._upsample(out_det, imgs.size())
            loss_det = self.det_head.loss(
                out_det, gt_texts, gt_kernels, training_masks,
                gt_instances, gt_bboxes)
            outputs.update(loss_det)
        else:
            out_det = self._upsample(out_det, imgs.size(), cfg.test_cfg.scale)
            res_det = self.det_head.get_results(out_det, img_metas, cfg)
            outputs.update(res_det)

        return outputs
