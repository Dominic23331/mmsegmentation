# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from torch.nn.modules.utils import _pair

from ..utils.vit_seg_modeling_resnet_skip import ResNetV2
from .vit import TransformerEncoderLayer


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
