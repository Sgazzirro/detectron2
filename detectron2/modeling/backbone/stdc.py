# Copyright (c) Facebook, Inc. and its affiliates.
import math
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from detectron2.modeling import Backbone, BACKBONE_REGISTRY

from detectron2.layers import ShapeSpec

__all__ = ["ConvX", "STDCNet1446"]

class ConvX(nn.Module):
    # * Basic Block of STDC backbone (Conv - BNorm - ReLU)
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class CatBottleneck(nn.Module):
    # * Basic Macro-Block of STDC backbone
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        # Block num is the number of basic blocks inside the macro. Always larger thank one
        # If stride = 1, we are inside an inner macro block
        # If stride = 2, we are at the last macro block
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

class STDCNet1446(Backbone):
    """
    Abstract base class for network backbones.
    """

    def __init__(self, base, layers, block_num, use_conv_last, pretrain_model, num_classes=1000, dropout=0.2,):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super(STDCNet1446, self).__init__()
        # * Here we define the structure of a block:
        # * In our pretrained versione, block are of type "Cat bottleneck"
        block = CatBottleneck
        self.features = self._make_layers(base, layers, block_num, block)
        self.use_conv_last = use_conv_last

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
            self._freeze_module()
        else:
            self.init_params()



    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def forward(self, x):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        # Create a dictionary with keys "STDC2", "STDC4", etc
        feat_dict = {"x2": feat2, "x4": feat4, "x8": feat8, "x16": feat16, "x32": feat32}
        return feat_dict

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        names = ["x2", "x4", "x8", "x16", "x32"]
        out_channels = [32, 64, 256, 512, 1024]
        return {
            name: ShapeSpec(
                channels=out_channels[i]
            )
            for i,name in enumerate(names)
        }

    def _freeze_module(self):
        """
        Freezes the entire module by disabling gradient computation for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = False

    def _make_layers(self, base, layers, block_num, block):
        # * This function creates all the backbone
        # * Feature maps are:
        # * (3, Base // 2) 3 x 3 stride 2
        # * (Base // 2) 3 x 3 stride 2
        # * In my pretrained versione the Base channels are 64
        # * (3, 32) -> (32, 64)
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    # * At the very first block, I'm creating a (64, 256) stride 2
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    # * If I'm at the end of a block (but not the very first), I'm doubling channels and having a stride 2
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    # * Otherwise, same channels with stride 1
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)


@BACKBONE_REGISTRY.register()
def build_stdc_backbone(cfg):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    base = 64
    layers = [4, 5, 3]
    block_num = 4
    return STDCNet1446(base, layers, block_num, pretrain_model=pretrain_path, use_conv_last=False)