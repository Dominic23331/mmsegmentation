import warnings
import math
import copy

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from mmengine.model import BaseModule, ModuleList, Sequential
from mmseg.registry import MODELS


class Conv2dReLU(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        self.relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class DecoderBlock(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
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


class DecoderCup(BaseModule):
    def __init__(self,
                 hidden_size=768,
                 head_channels=512,
                 decoder_channels=(256, 128, 64, 16),
                 n_skip=3,
                 skip_channels=[512, 256, 64, 16]
                 ):
        super().__init__()
        self.n_skip = n_skip
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4 - n_skip):
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch
            in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = ModuleList(blocks)

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
        return x


class SegmentationHead(Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 upsampling=1):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling)\
            if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


@MODELS.register_module()
class TransUnetHead(BaseModule):
    def __init__(self,
                 num_classes,
                 hidden_size=768,
                 head_channels=512,
                 decoder_channels=(256, 128, 64, 16),
                 n_skip=3,
                 skip_channels=[512, 256, 64, 16],
                 head_kernal_size=3,
                 head_upsample_scale_factor=1):
        super().__init__()
        self.decoder = DecoderCup(hidden_size=hidden_size,
                                  head_channels=head_channels,
                                  decoder_channels=decoder_channels,
                                  n_skip=n_skip,
                                  skip_channels=skip_channels)

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=head_kernal_size,
            upsampling=head_upsample_scale_factor
        )

    def forward(self, x):
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x
