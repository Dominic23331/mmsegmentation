import warnings
import math
import copy

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS

from ..utils.vit_seg_modeling_resnet_skip import ResNetV2



class Embeddings(BaseModule):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,
                 img_size=224,
                 patches_size=16,
                 in_channels=3,
                 hidden_size=768,
                 dropout_rate=0.1,
                 resnet_layers=None,
                 resnet_width_factor=None,
                 grid=None
                 ):
        super().__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if grid is not None:  # ResNet
            grid_size = grid
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) \
                        * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(patches_size)
            n_patches = (img_size[0] // patch_size[0]) \
                        * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=resnet_layers,
                                         width_factor=resnet_width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Mlp(BaseModule):
    def __init__(self,
                 hidden_size=768,
                 mlp_dim=3072,
                 dropout_rate=0.1
                 ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(BaseModule):
    def __init__(self,
                 num_heads=12,
                 hidden_size=768,
                 attention_dropout_rate=0.
                 ):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads \
                             * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] \
                      + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores \
                           / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2]\
                                  + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(BaseModule):
    def __init__(self,
                 hidden_size=768,
                 mlp_dim=3072,
                 num_heads=12,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.
                 ):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size=hidden_size,
                       mlp_dim=mlp_dim,
                       dropout_rate=dropout_rate)
        self.attn = Attention(num_heads=num_heads,
                              hidden_size=hidden_size,
                              attention_dropout_rate=attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(BaseModule):
    def __init__(self,
                 num_layers=12,
                 hidden_size=768,
                 mlp_dim=3072,
                 num_heads=12,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.
                 ):
        super().__init__()
        self.layer = ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size=hidden_size,
                          mlp_dim=mlp_dim,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate,
                          attention_dropout_rate=attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


@MODELS.register_module()
class TransUnet(BaseModule):
    def __init__(self,
                 img_size=224,
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
                 grid=None
                 ):
        super().__init__()
        self.embeddings = Embeddings(img_size=img_size,
                                     patches_size=patches_size,
                                     in_channels=in_channels,
                                     hidden_size=hidden_size,
                                     dropout_rate=dropout_rate,
                                     resnet_layers=resnet_layers,
                                     resnet_width_factor=resnet_width_factor,
                                     grid=grid)
        self.encoder = Encoder(num_layers=num_layers,
                               hidden_size=hidden_size,
                               mlp_dim=mlp_dim,
                               num_heads=num_heads,
                               dropout_rate=dropout_rate,
                               attention_dropout_rate=attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features
