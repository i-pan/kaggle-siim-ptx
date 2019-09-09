from .resnext import ResNeXt, ResNet
from mmdet.models.backbones import HRNet

import pretrainedmodels
import pretrainedmodels.utils
import torch 

from torch import nn
from torch.nn import functional as F

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
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
    def __init__(self, dim, reduction=4):
        super(scSE, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim, reduction)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

strides_and_dilations = {
    16 : {
        'strides':   (1, 2, 2, 1),
        'dilations': (1, 1, 1, 2),
        'mg_rates':  (1, 2, 4)
    },
    8  : {
        'strides':   (1, 2, 1, 1),
        'dilations': (1, 1, 2, 4),
        'mg_rates':  (1, 2, 4)
    },
    'jpu': {
        'strides':   (1, 2, 2, 2),
        'dilations': (1, 1, 1, 1),
        'mg_rates':  (1, 1, 1)    
    }
}

def build_mmdet_res_model(output_stride, jpu,
                          use_scse, use_maxpool,
                          backbone, pretrained, preprocessing, 
                          encoder_channels, group_norm):
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights(pretrained)
    model.input_range = preprocessing['input_range']
    model.mean = preprocessing['mean']
    model.std  = preprocessing['std']
    if group_norm:
        low_level_list = [model.conv1, model.gn1, model.relu]
    else:
        low_level_list = [model.conv1, model.bn1, model.relu]
    if use_maxpool: 
        low_level_list.append(model.maxpool)
    low_level_list.append(model.layer1)
    if use_scse: 
        low_level_list.append(scSE(encoder_channels[0]))
    low_level = nn.Sequential(*low_level_list)
    if jpu:
        # If using JPU, need to return a list of encoder blocks
        # so that final 3 encoder feature maps can be returned
        if use_scse:
            encoder = [nn.Sequential(model.layer2, scSE(encoder_channels[1])),
                       nn.Sequential(model.layer3, scSE(encoder_channels[2])),
                       nn.Sequential(model.layer4, scSE(encoder_channels[3]))]
        else:
            encoder = [model.layer2, model.layer3, model.layer4]
    else:
        if use_scse:
            encoder = nn.Sequential(model.layer2, scSE(encoder_channels[1]),
                                    model.layer3, scSE(encoder_channels[2]),
                                    model.layer4, scSE(encoder_channels[3]))
        else:
            encoder = nn.Sequential(model.layer2,
                                    model.layer3,
                                    model.layer4)
    return (encoder, low_level), encoder_channels, model

##########################
# GROUPNORM + WEIGHT STD #
##########################

def resnet50_gn_ws(output_stride=16, 
                   jpu=True, 
                   use_scse=True, 
                   use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnet50_gn_ws',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnet101_gn_ws(output_stride=16, 
                    jpu=True, 
                    use_scse=True, 
                    use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnet101_gn_ws',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnext50_gn_ws(output_stride=16, 
                    jpu=True, 
                    use_scse=True, 
                    use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnext50_32x4d_gn_ws',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnext101_gn_ws(output_stride=16, 
                     jpu=True, 
                     use_scse=True, 
                     use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='ConvWS')
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnext101_32x4d_gn_ws',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

#############
# GROUPNORM #
#############

def resnet50_gn(output_stride=16, 
                jpu=True, 
                use_scse=True, 
                use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [102.9801,115.9465,122.7717], 'std': [1.0,1.0,1.0]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://detectron/resnet50_gn',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnet101_gn(output_stride=16, 
                 jpu=True, 
                 use_scse=True, 
                 use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [102.9801,115.9465,122.7717], 'std': [1.0,1.0,1.0]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://detectron/resnet101_gn',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnext50_gn(output_stride=16, 
                 jpu=True, 
                 use_scse=True, 
                 use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
    encoder_channels = [256, 512, 1024, 2048]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnext50_32x4d_gn',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def resnext101_gn(output_stride=16, 
                  jpu=True, 
                  use_scse=True, 
                  use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    encoder_channels = [256, 512, 1024, 2048]
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnext101_32x4d_gn',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod

def dcn_resnext101_gn(output_stride=16, 
                      jpu=True, 
                      use_scse=True, 
                      use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        stage_with_dcn=(False, True, True, True),
        dcn=dict(modulated=True, groups=32, deformable_groups=1, fallback_on_stride=False),
        frozen_stages=0,
        style='pytorch',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
    encoder_channels = [256, 512, 1024, 2048] 
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://jhu/resnext101_32x4d_gn',
        preprocessing, encoder_channels, group_norm=True
    )
    return (enc[0], enc[1]), ch, mod   


############# 
# BATCHNORM #
#############
# BatchNorm is NOT frozen
# i.e., requires_grad + train mode 

def resnext101_64x4d(output_stride=16, 
                     jpu=True, 
                     use_scse=True, 
                     use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_eval=False
    )
    encoder_channels = [256, 512, 1024, 2048] 
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://resnext101_64x4d',
        preprocessing, encoder_channels, group_norm=False
    )
    return (enc[0], enc[1]), ch, mod    

def resnext101_32x4d(output_stride=16, 
                     jpu=True, 
                     use_scse=True, 
                     use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        frozen_stages=0,
        style='pytorch',
        norm_eval=False
    )
    encoder_channels = [256, 512, 1024, 2048] 
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://resnext101_32x4d',
        preprocessing, encoder_channels, group_norm=False
    )
    return (enc[0], enc[1]), ch, mod   

# TODO: keeps throwing an error
def dcn_resnext101_32x4d(output_stride=16, 
                         jpu=True, 
                         use_scse=True, 
                         use_maxpool=True):
    os = output_stride if not jpu else 'jpu'
    backbone = dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        mg_rates=strides_and_dilations[os]['mg_rates'],
        strides=strides_and_dilations[os]['strides'],
        dilations=strides_and_dilations[os]['dilations'],
        stage_with_dcn=(False, True, True, True),
        dcn=dict(modulated=True, groups=32, deformable_groups=1, fallback_on_stride=False),
        frozen_stages=0,
        style='pytorch',
        norm_eval=False
    )
    encoder_channels = [256, 512, 1024, 2048] 
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    enc, ch, mod = build_mmdet_res_model(
        output_stride, jpu, use_scse, use_maxpool, 
        backbone, 'open-mmlab://resnext101_32x4d',
        preprocessing, encoder_channels, group_norm=False
    )
    return (enc[0], enc[1]), ch, mod   

#########
# HRNET #
#########

# HRNet will not work with JPU
# HRNet does not have dilations

class HREncoder(nn.Module):
    def __init__(self, stages, stage_cfgs, transitions):
        super(HREncoder, self).__init__()
        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]
        self.stage2_cfg = stage_cfgs[0]
        self.stage3_cfg = stage_cfgs[1]
        self.stage4_cfg = stage_cfgs[2]
        self.transition1 = transitions[0]
        self.transition2 = transitions[1]
        self.transition3 = transitions[2]
    def forward(self, x):
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['num_branches']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['num_branches']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear')
        x = torch.cat([x[0], x1, x2, x3], 1)
        return x

def hrnetv2_w18(output_stride=None,
                jpu=None,
                use_scse=True,
                use_maxpool=None):
    backbone=dict(
        type='HRNet',
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144))))
    encoder_channels = [256, None, None, 270]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://msra/hrnetv2_w18')
    model.input_range = preprocessing['input_range']
    model.mean = preprocessing['mean']
    model.std  = preprocessing['std']
    low_level_list = [model.conv1, model.norm1, model.relu, model.conv2, model.norm2, model.relu, model.layer1]
    if use_scse:
        low_level_list.append(scSE(encoder_channels[0]))
    low_level = nn.Sequential(*low_level_list)
    encoder = HREncoder(stages=[model.stage2,model.stage3,model.stage4],
                        stage_cfgs=[model.stage2_cfg,model.stage3_cfg,model.stage4_cfg],
                        transitions=[model.transition1,model.transition2,model.transition3])
    if use_scse:
        encoder = nn.Sequential(encoder, scSE(encoder_channels[3]))
    return (encoder, low_level), encoder_channels, model

def hrnetv2_w32(output_stride=None,
                jpu=None,
                use_scse=True,
                use_maxpool=None):
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))))
    encoder_channels = [256, None, None, 480]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://msra/hrnetv2_w32')
    model.input_range = preprocessing['input_range']
    model.mean = preprocessing['mean']
    model.std  = preprocessing['std']
    low_level_list = [model.conv1, model.norm1, model.relu, model.conv2, model.norm2, model.relu, model.layer1]
    if use_scse:
        low_level_list.append(scSE(encoder_channels[0]))
    low_level = nn.Sequential(*low_level_list)
    encoder = HREncoder(stages=[model.stage2,model.stage3,model.stage4],
                        stage_cfgs=[model.stage2_cfg,model.stage3_cfg,model.stage4_cfg],
                        transitions=[model.transition1,model.transition2,model.transition3])
    if use_scse:
        encoder = nn.Sequential(encoder, scSE(encoder_channels[3]))
    return (encoder, low_level), encoder_channels, model


def hrnetv2_w40(output_stride=None,
                jpu=None,
                use_scse=True,
                use_maxpool=None):
    backbone=dict(
        type='HRNet',
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320))))
    encoder_channels = [256, None, None, 600]
    preprocessing = {'input_range': [0,255], 'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}
    model = backbone.pop('type')
    model = eval(model)(**backbone)
    model.init_weights('open-mmlab://msra/hrnetv2_w40')
    model.input_range = preprocessing['input_range']
    model.mean = preprocessing['mean']
    model.std  = preprocessing['std']
    low_level_list = [model.conv1, model.norm1, model.relu, model.conv2, model.norm2, model.relu, model.layer1]
    if use_scse:
        low_level_list.append(scSE(encoder_channels[0]))
    low_level = nn.Sequential(*low_level_list)
    encoder = HREncoder(stages=[model.stage2,model.stage3,model.stage4],
                        stage_cfgs=[model.stage2_cfg,model.stage3_cfg,model.stage4_cfg],
                        transitions=[model.transition1,model.transition2,model.transition3])
    if use_scse:
        encoder = nn.Sequential(encoder, scSE(encoder_channels[3]))
    return (encoder, low_level), encoder_channels, model
