# From https://github.com/jfzhang95/pytorch-deeplab-xception

import peepdom.backbones_deeplab as backbones

import torch

from torch import nn
from torch.nn import functional as F

from torch.nn.modules.batchnorm import _BatchNorm

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels):
        super(GroupNorm32, self).__init__(num_channels=num_channels, num_groups=32)

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
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu(x)

        return self.dropout(x)

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

    def forward(self, x, low_level_feat, classifier=False):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm1(low_level_feat)
        low_level_feat = self.elu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feat), dim=1)
        decoder_output = x
        x = self.last_conv(x)
        if classifier:
            return x, decoder_output
        else: 
            return x


class DeepLab(nn.Module):
    def __init__(self, backbone, 
        output_stride=16,
        group_norm=True, 
        classifier=False, 
        dropout=dict(
            spp=0.5,
            cls=0.2,
            dc0=0.5,
            dc1=0.1
        ),
        num_classes=2,
        norm_eval=False):
        super(DeepLab, self).__init__()

        layers, channels, backbone = getattr(backbones, backbone)(output_stride=output_stride)

        self.input_range = backbone.input_range
        self.mean = backbone.mean 
        self.std = backbone.std

        self.classifier = classifier

        # default is freeze BatchNorm
        self.norm_eval = norm_eval

        norm_layer = GroupNorm32 if group_norm else nn.BatchNorm2d
        
        self.backbone = layers[0]
        self.low_level = layers[1]

        self.aspp_planes = 256

        if output_stride == 16:
            aspp_dilations = (1,  6, 12, 18)
        elif output_stride == 8:
            aspp_dilations = (1, 12, 24, 36)

        self.spp = ASPP(aspp_dilations, inplanes=channels[1], planes=self.aspp_planes, dropout=dropout['spp'], norm_layer=norm_layer)
        self.decoder = Decoder(num_classes, self.aspp_planes, channels[0], 64, (dropout['dc0'], dropout['dc1']), norm_layer)
        self.train_mode = True

        # classifier branch
        if classifier: 
            self.logit_image = nn.Sequential(nn.Dropout(dropout['cls']), nn.Linear(channels[1]+self.aspp_planes+64, num_classes))

    def forward(self, x_input):
        low_level_feat = self.low_level(x_input)
        features = self.backbone(x_input)

        x = self.spp(features)
        if self.classifier:
            x, decoder_output = self.decoder(x, low_level_feat, classifier=self.classifier)
        else:
            x = self.decoder(x, low_level_feat, classifier=self.classifier)
        out_size = x_input.size()[2:]
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)

        # classifier branch
        if self.classifier: 
            features = features.mean([2, 3])
            decoder_output = decoder_output.mean([2,3])
            features = torch.cat((features, decoder_output), dim=1)
            c = self.logit_image(features)
            return x, c
        else:
            return x

    def train(self, mode=True):
        super(DeepLab, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self


