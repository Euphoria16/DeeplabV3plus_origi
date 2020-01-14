# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from graphs.models.ResNet101 import resnet101
from graphs.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import sys
sys.path.append(os.path.abspath('..'))

from graphs.models.encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()



    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)

        x_4 = F.interpolate(x, size=low_level_feature.size()[2:4], mode='bilinear' ,align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)


        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)


        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)


        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DeepLab(nn.Module):
    def __init__(self, output_stride, class_num, pretrained, bn_momentum=0.1, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(class_num, bn_momentum)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low0,low1,low2 = self.Resnet101(input)

        x = self.encoder(x)
        predict = self.decoder(x, low1)
        output= F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels1, in_channels2,out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels2 , in_channels2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels1+in_channels2, out_channels)

    def forward(self, x1, x2):
        # print('before upsample',x1.size())
        x1 = self.up(x1)
        # print('after upsample', x1.size())
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Unet_decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1,
                 bilinear=True):
        super(Unet_decoder, self).__init__()

        # self.cascade_num = cascade_num
        self.up1= Up(48,256,256,bilinear=bilinear)#to 1/8
        self.up2 = Up(48 ,256, 128, bilinear=bilinear)# to 1/4
        self.up3 = Up(32,128,64,bilinear=bilinear)# to 1/2
        self.up4 = Up(32,64,64,bilinear=bilinear)
        self.OutConv=OutConv(64,class_num)

        self.conv1 = nn.Conv2d(512, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(32, momentum=bn_momentum)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.bn4 = SynchronizedBatchNorm2d(32, momentum=bn_momentum)

        self.inc = DoubleConv(3, 64)

        self._init_weight()

    def forward(self, x,low0,low1,low2,input):
        # x,low0,low1,low2=self.Resnet101(input)#1/2+64,1/4+256,1/8+512
        # inc1=self.inc(input)
        # print('inc1',inc1.size())#64,1/1
        # x = self.encoder(x)

        inc1 = self.inc(input)

        low2 = self.conv1(low2)
        low2 = self.bn1(low2)
        low2 = self.relu(low2)
        x =self.up1(x,low2)#([8, 256, 65, 65])
        # print('low2 size',low2.size())
        # print('up1 size', x.size())

        low1 = self.conv2(low1)
        low1 = self.bn2(low1)
        low1 = self.relu(low1)
        x =self.up2(x,low1)#([8, 128, 129, 129])
        # print('low1 size',low1.size())
        # print('up2 size', x.size())

        low0 = self.conv3(low0)
        low0 = self.bn3(low0)
        low0 = self.relu(low0)
        x =self.up3(x,low0)
        # print('low0 size',low0.size())
        # print('up3 size',x.size())

        inc1 = self.conv4(inc1)
        inc1 = self.bn4(inc1)
        inc1 = self.relu(inc1)
        x =self.up4(x,inc1)
        # print('inc1 size',inc1.size())
        # print('up4 size',x.size())

        seg=self.OutConv(x)
        # print('seg size',seg.size())

        # seg= F.interpolate(seg,input.size()[2:], mode='bilinear' ,align_corners=True)
        return seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLab_Unet(nn.Module):
    def __init__(self, output_stride, class_num, pretrained, bn_momentum=0.1, freeze_bn=False,bilinear=True):
        super(DeepLab_Unet, self).__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Unet_decoder(class_num, bn_momentum,
                 bilinear)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low0,low1,low2 = self.Resnet101(input)
        x = self.encoder(x)
        predict = self.decoder(x, low0,low1,low2,input)
        # output= F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return predict

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


class Up_input(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, input):
        # print('before upsample',x1.size())
        x1 = self.up(x1)
        # print('after upsample', x1.size())
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, input, x1], dim=1)
        return self.conv(x)
class Unet_decoder_input(nn.Module):
    def __init__(self, output_stride, class_num, pretrained, bn_momentum=0.1, freeze_bn=False,
                 bilinear=True):
        super(Unet_decoder_input, self).__init__()
        self.Resnet101 = resnet101(bn_momentum, pretrained, output_stride)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.inc = DoubleConv(3, 64)
        # self.cascade_num = cascade_num
        self.up1= Up_input(512+256+3,256,bilinear=bilinear)#to 1/8
        self.up2 = Up_input(256 + 256+3, 128, bilinear=bilinear)# to 1/4
        self.up3 = Up_input(128+64+3,64,bilinear=bilinear)# to 1/2
        self.up4 = Up_input(64 + 64 +3, 64, bilinear=bilinear)
        self.OutConv=OutConv(64,class_num)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x,low0,low1,low2=self.Resnet101(input)
        inc1 = self.inc(input)
        # input_4 = F.interpolate(input, low0.size()[2:], mode='bilinear', align_corners=True)
        # input_2 = F.interpolate(input, low0.size()[2:], mode='bilinear', align_corners=True)
        x = self.encoder(x)

        input_16 = F.interpolate(input, low2.size()[2:4], mode='bilinear', align_corners=True)
        # x =self.up1(torch.cat((x,input_16),dim=1))#1/8
        x=self.up1(x,low2,input_16)
        # print('up1',x.size())

        input_8 = F.interpolate(input, low1.size()[2:4], mode='bilinear', align_corners=True)
        # x =self.up2(low1,torch.cat((x,input_8),dim=1))#1/4
        x=self.up2(x,low1,input_8)
        # print('up2', x.size())


        input_4 = F.interpolate(input, low0.size()[2:4], mode='bilinear', align_corners=True)
        # x =self.up3(low0,torch.cat((x,input_4),dim=1))#1/2
        x=self.up3(x,low0,input_4)
        # print('up3', x.size())

        x=self.up4(x,inc1,input)
        seg=self.OutConv(x)
        # print('up4', x.size())
        # print('seg size',seg.size())


        # input_2 = F.interpolate(input, x.size()[2:4], mode='bilinear', align_corners=True)
        # seg=self.up(torch.cat((x,input_2),dim=1))#1/2

        # seg= F.interpolate(seg,input.size()[2:], mode='bilinear' ,align_corners=True)
        return seg

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
if __name__ =="__main__":
    model = DeepLab(output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    # print(model)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # summary(model, (3, 513, 513))
    # for m in model.named_modules():
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
