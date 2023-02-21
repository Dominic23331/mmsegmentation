# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class Conv2dReLU(Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)


class DecoderBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels=0,
                 use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class TransUnetHead(BaseDecodeHead):

    def __init__(self,
                 hidden_size,
                 n_skip,
                 skip_channels=None,
                 head_channels=512,
                 dropout_ratio=0.,
                 loss_decode=[
                     dict(
                         type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=0.5),
                     dict(
                         type='FocalLoss', use_sigmoid=False, loss_weight=0.5)
                 ],
                 **kwargs):
        super().__init__(**kwargs)
        self.n_skip = n_skip
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)
        in_channels = [head_channels] + list(self.in_channels[:-1])

        if n_skip != 0:
            for i in range(4 - n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(
                in_channels, self.in_channels, skip_channels)
        ]
        self.blocks = ModuleList(blocks)

        self.conv_seg.kernel_size = 3
        self.conv_seg.stride = 1
        self.conv_seg.padding = 1

    def forward(self, inputs):
        hidden_states, features = inputs
        B, n_patch, hidden = hidden_states.size()
        h, w = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        out = self.cls_seg(x)
        return out
