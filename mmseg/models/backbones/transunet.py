# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.registry import MODELS
from torch.nn.modules.utils import _pair

from ..utils.vit_seg_modeling_resnet_skip import ResNetV2
from .vit import TransformerEncoderLayer


class TransUnetEncoder(BaseModule):

    def __init__(self,
                 img_size=(224, 224),
                 patches_size=16,
                 in_channels=3,
                 hidden_size=768,
                 num_layers=12,
                 mlp_dim=3072,
                 num_heads=12,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.,
                 resnet_layers=None,
                 resnet_width_factor=None,
                 grid=None):
        super().__init__()
        if isinstance(img_size, int):
            img_size = _pair(img_size)

        if grid is not None:
            self.build_resnet(resnet_layers, resnet_width_factor)
            patches_size = (img_size[0] // 16 // grid[0],
                            img_size[1] // 16 // grid[1])
            patch_size_real = (patches_size[0] * 16, patches_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1])
            in_channels = self.resnet.width * 16
            self.use_resnet = True
        else:
            patches_size = _pair(patches_size)
            n_patches = (img_size[0] // patches_size[0]) * (
                img_size[1] // patches_size[1])
            self.use_resnet = False

        self.patch_embd = PatchEmbed(
            in_channels=in_channels,
            embed_dims=hidden_size,
            kernel_size=patches_size,
            stride=patches_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))

        self.dropout = nn.Dropout(dropout_rate)

        self.TFlayers = ModuleList()
        for i in range(num_layers):
            self.TFlayers.append(
                TransformerEncoderLayer(
                    embed_dims=hidden_size,
                    num_heads=num_heads,
                    feedforward_channels=mlp_dim,
                    drop_rate=dropout_rate,
                    attn_drop_rate=attention_dropout_rate))

    def build_resnet(self, block_units, width_factor):
        self.resnet = ResNetV2(block_units, width_factor)

    def forward(self, x):
        if self.use_resnet:
            x, features = self.resnet(x)
        else:
            features = None

        x, _ = self.patch_embd(x)
        x = x + self.position_embedding
        x = self.dropout(x)

        for layer in self.TFlayers:
            x = layer(x)
        return (x, features)


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


class TransUnetDecoder(BaseModule):

    def __init__(self,
                 hidden_size=768,
                 n_skip=3,
                 skip_channels=[512, 256, 64, 16],
                 head_channels=512,
                 decoder_channels=(256, 128, 64, 16)):
        super().__init__()
        self.n_skip = n_skip
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True)
        in_channels = [head_channels] + list(decoder_channels[:-1])

        if n_skip != 0:
            for i in range(4 - n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(
                in_channels, decoder_channels, skip_channels)
        ]
        self.blocks = ModuleList(blocks)

    def forward(self, hidden_states, features):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        dec_out = [x]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            dec_out.append(x)

        return dec_out


@MODELS.register_module()
class TransUnet(BaseModule):

    def __init__(self,
                 img_size=(224, 224),
                 patches_size=16,
                 in_channels=3,
                 hidden_size=768,
                 num_layers=12,
                 mlp_dim=3072,
                 num_heads=12,
                 head_channels=512,
                 decoder_channels=(256, 128, 64, 16),
                 n_skip=3,
                 skip_channels=[512, 256, 64, 16],
                 dropout_rate=0.1,
                 attention_dropout_rate=0.,
                 resnet_layers=None,
                 resnet_width_factor=None,
                 grid=None):
        super().__init__()
        self.encoder = TransUnetEncoder(img_size, patches_size, in_channels,
                                        hidden_size, num_layers, mlp_dim,
                                        num_heads, dropout_rate,
                                        attention_dropout_rate, resnet_layers,
                                        resnet_width_factor, grid)

        self.decoder = TransUnetDecoder(hidden_size, n_skip, skip_channels,
                                        head_channels, decoder_channels)

    def forward(self, x):
        x, features = self.encoder(x)
        out = self.decoder(x, features)
        return out
