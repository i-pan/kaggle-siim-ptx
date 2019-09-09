import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnext import ResNeXt, ResNet

class SpatialAttention2d(nn.Module):
    def __init__(self, channel, conv_layer):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class GAB(nn.Module):
    def __init__(self, input_dim, conv_layer, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class scSE(nn.Module):
    def __init__(self, dim, conv_layer, reduction=4):
        super(scSE, self).__init__()
        self.satt = SpatialAttention2d(dim, conv_layer)
        self.catt = GAB(dim, conv_layer, reduction)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

def resnet50_gn_ws(output_stride=16, use_scse=True):
    if output_stride == 16:
        strides = (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
    elif output_stride == 8:
        strides = (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=(1, 2, 4),
        strides=strides,
        dilations=dilations,
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')    
    )
    #
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://jhu/resnet50_gn_ws')
    model.input_range = [0, 255]
    model.mean = [102.9801, 115.9465, 122.7717]
    model.std  = [1.0, 1.0, 1.0]
    #
    low_level = nn.Sequential(model.conv1, model.gn1, model.relu, model.maxpool, model.layer1)
    if use_scse:
        encoder = nn.Sequential(low_level,
                                scSE(256, nn.Conv2d),
                                model.layer2, 
                                scSE(512, nn.Conv2d),
                                model.layer3,
                                scSE(1024, nn.Conv2d),
                                model.layer4,
                                scSE(2048, nn.Conv2d))
    else:
        encoder = model
    return (encoder, low_level), [256, 2048], model

def resnet101_gn_ws(output_stride=16, use_scse=True):
    if output_stride == 16:
        strides = (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
    elif output_stride == 8:
        strides = (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
    backbone = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=(1, 2, 4),
        strides=strides,
        dilations=dilations,
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    #
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://jhu/resnet101_gn_ws')
    model.input_range = [0, 255]
    model.mean = [123.675, 116.28, 103.53]
    model.std  = [58.395, 57.12, 57.375]
    #
    low_level = nn.Sequential(model.conv1, model.gn1, model.relu, model.maxpool, model.layer1)
    if use_scse:
        encoder = nn.Sequential(low_level,
                                scSE(256, nn.Conv2d),
                                model.layer2, 
                                scSE(512, nn.Conv2d),
                                model.layer3,
                                scSE(1024, nn.Conv2d),
                                model.layer4,
                                scSE(2048, nn.Conv2d))
    else:
        encoder = model
    return (encoder, low_level), [256, 2048], model


def resnext50_gn_ws(output_stride=16, use_scse=True):
    if output_stride == 16:
        strides = (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
    elif output_stride == 8:
        strides = (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
    backbone = dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=(1, 2, 4),
        strides=strides,
        dilations=dilations,
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    #
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://jhu/resnext50_32x4d_gn_ws')
    model.input_range = [0, 255]
    model.mean = [123.675, 116.28, 103.53]
    model.std  = [58.395, 57.12, 57.375]
    #
    low_level = nn.Sequential(model.conv1, model.gn1, model.relu, model.maxpool, model.layer1)
    if use_scse:
        encoder = nn.Sequential(low_level,
                                scSE(256, nn.Conv2d),
                                model.layer2, 
                                scSE(512, nn.Conv2d),
                                model.layer3,
                                scSE(1024, nn.Conv2d),
                                model.layer4,
                                scSE(2048, nn.Conv2d))
    else:
        encoder = model
    return (encoder, low_level), [256, 2048], model

def resnext101_gn_ws(output_stride=16, use_scse=True):
    if output_stride == 16:
        strides = (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
    elif output_stride == 8:
        strides = (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=(1, 2, 4),
        strides=strides,
        dilations=dilations,
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    #
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://jhu/resnext101_32x4d_gn_ws')
    model.input_range = [0, 255]
    model.mean = [123.675, 116.28, 103.53]
    model.std  = [58.395, 57.12, 57.375]
    #
    low_level = nn.Sequential(model.conv1, model.gn1, model.relu, model.maxpool, model.layer1)
    if use_scse:
        encoder = nn.Sequential(low_level,
                                scSE(256, nn.Conv2d),
                                model.layer2, 
                                scSE(512, nn.Conv2d),
                                model.layer3,
                                scSE(1024, nn.Conv2d),
                                model.layer4,
                                scSE(2048, nn.Conv2d))
    else:
        encoder = model
    return (encoder, low_level), [256, 2048], model