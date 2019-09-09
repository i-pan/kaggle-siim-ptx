# From https://github.com/jfzhang95/pytorch-deeplab-xception

import peepdom.backbones_deeplab_jpu as backbones
try:
    from model.encoding import *
except:
    pass

from torch import nn
from torch.nn import functional as F

import torch

# Wrapper for GroupNorm with 32 channels
class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels):
        super(GroupNorm32, self).__init__(num_channels=num_channels, num_groups=32)

########
# ASPP #
########

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.norm = norm_layer(planes)
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.elu = nn.ELU(True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.norm(x)
        return self.elu(x)

class ASPP(nn.Module):
    def __init__(self, dilations, inplanes, planes, norm_layer, dropout=0.5):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, planes, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, planes, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, planes, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.norm1 = norm_layer(planes)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                             norm_layer(planes),
                                             nn.ELU(True))
        self.conv1 = nn.Conv2d(5 * planes, planes, 1, bias=False)
        self.elu = nn.ELU(True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu(x)

        return self.dropout(x)

#######
# FPA #
#######

# From phalanx
class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer):
        super(FPAv2, self).__init__()
        self.glob    = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     norm_layer(input_dim))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     norm_layer(output_dim))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     norm_layer(input_dim))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     norm_layer(output_dim))

        self.conv1   = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                     norm_layer(output_dim))

    def forward(self, x):
        x_glob = self.glob(x)
        x_glob = F.interpolate(x_glob, scale_factor=int(x.size()[-1] / x_glob.size()[-1]), mode='bilinear')  # 256, 16, 16

        d2 = F.elu(self.down2_1(x)) 
        d3 = F.elu(self.down3_1(d2)) 
        d2 = F.elu(self.down2_2(d2)) 
        d3 = F.elu(self.down3_2(d3)) 
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear')  # 256, 8, 8
        d2 = d2 + d3
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear')  # 256, 16, 16

        x = F.elu(self.conv1(x)) 
        x = x * d2
        x = x + x_glob

        return x

#######
# JPU #
#######

# From https://github.com/wuhuikai/FastFCN/
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class JPU16(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None):
        super(JPU16, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], size=(h, w), mode='bilinear')
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat

class JPU08(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None):
        super(JPU08, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], size=(h, w), mode='bilinear')
        feats[-3] = F.interpolate(feats[-3], size=(h, w), mode='bilinear')
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat

#######
# PSP #
#######
# From https://github.com/Lextal/pspnet-pytorch

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

# Decoder for DeepLab
class Decoder(nn.Module):
    def __init__(self, num_classes, spp_inplanes, low_level_inplanes, inplanes, dropout, norm_layer):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, inplanes, 1, bias=False)
        self.norm1 = norm_layer(inplanes)
  
        self.elu = nn.ELU(True)
        self.last_conv = nn.Sequential(nn.Conv2d(spp_inplanes + inplanes, spp_inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(spp_inplanes),
                                       nn.ELU(True),
                                       nn.Dropout2d(dropout[0]),
                                       nn.Conv2d(spp_inplanes, spp_inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(spp_inplanes),
                                       nn.ELU(True),
                                       nn.Dropout2d(dropout[1]),
                                       nn.Conv2d(spp_inplanes, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm1(low_level_feat)
        low_level_feat = self.elu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear')
        x = torch.cat((x, low_level_feat), dim=1)
        decoder_output = x
        x = self.last_conv(x)
        return x

class DeepLab(nn.Module):
    def __init__(self, backbone, 
        output_stride=16,
        group_norm=True, 
        dropout=dict(
            spp=0.5,
            dc0=0.5,
            dc1=0.1
        ),
        num_classes=2,
        center='aspp',
        jpu=True,
        norm_eval=False,
        use_maxpool=True):
        super(DeepLab, self).__init__()

        layers, channels, backbone = getattr(backbones, backbone)(output_stride=output_stride, jpu=jpu, use_maxpool=use_maxpool)

        self.input_range = backbone.input_range
        self.mean = backbone.mean 
        self.std = backbone.std

        # default is freeze BatchNorm
        self.norm_eval = norm_eval

        norm_layer = GroupNorm32 if group_norm else nn.BatchNorm2d
        
        if jpu:
            self.backbone1 = layers[0][0]
            self.backbone2 = layers[0][1]
            self.backbone3 = layers[0][2]
        else:
            self.backbone  = layers[0]
        self.low_level = layers[1]
        self.center_type = center
        self.use_jpu = jpu

        self.aspp_planes = 256
        self.output_stride = output_stride

        if output_stride == 16:
            aspp_dilations = (1,  6, 12, 18)
        elif output_stride == 8:
            aspp_dilations = (1, 12, 24, 36)

        center_input_channels = channels[-1]

        if center == 'fpa':
            self.center = FPAv2(center_input_channels, self.aspp_planes, norm_layer=norm_layer)
        elif center == 'aspp':
            self.center = ASPP(aspp_dilations, inplanes=center_input_channels, planes=self.aspp_planes, dropout=dropout['spp'], norm_layer=norm_layer)
        elif center == 'psp':
            self.center = PSPModule(center_input_channels, out_features=self.aspp_planes)
        elif center == 'enc':
            self.center = EncModule(center_input_channels, self.aspp_planes, norm_layer)
        if jpu:
            if output_stride == 16:
                self.jpu = JPU16(channels[2:], norm_layer=norm_layer, width=center_input_channels // 4)
            elif output_stride == 8: 
                self.jpu = JPU08(channels[1:], norm_layer=norm_layer, width=center_input_channels // 4)

        self.decoder = Decoder(num_classes, self.aspp_planes, channels[0], 64, (dropout['dc0'], dropout['dc1']), norm_layer)
        self.train_mode = True

    def forward(self, x_input):
        low_level_feat = self.low_level(x_input)
        if self.use_jpu:
            c2 = self.backbone1(low_level_feat)
            c3 = self.backbone2(c2)
            c4 = self.backbone3(c3)
        else:
            features = self.backbone(low_level_feat)

        if self.use_jpu:
            if self.output_stride == 16:
                x = self.center(self.jpu(c3, c4))
            elif self.output_stride == 8:
                x = self.center(self.jpu(c2, c3, c4))
        else:
            x = self.center(features)
        x = self.decoder(x, low_level_feat)
        out_size = x_input.size()[2:]
        x = F.interpolate(x, size=out_size, mode='bilinear')
        return x

    def train(self, mode=True):
        super(DeepLab, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


