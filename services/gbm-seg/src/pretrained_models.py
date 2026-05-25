from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from senet import *
from collections import OrderedDict
from fastai.vision.all import *


## https://github.com/ternaus/robot-surgery-segmentation/blob/master/models.py ##

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet_class(nn.Module):
    def __init__(self, model, num_classes=1, filters_mult=1):
        super().__init__()
        self.num_classes = num_classes
        filters = [filters_mult * 64, filters_mult * 128, filters_mult * 256, filters_mult * 512]
        resnet = model

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SE_class_LinkNet_network(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = model
        self.num_classes = num_classes

        self.encoder0 = self.class_model.layer0
        self.encoder1 = self.class_model.layer1
        self.encoder2 = self.class_model.layer2
        self.encoder3 = self.class_model.layer3
        self.encoder4 = self.class_model.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x):

        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SE_class_multi_LinkNet_network(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = model
        self.num_classes = num_classes

        self.encoder0 = self.class_model.layer0
        self.encoder1 = self.class_model.layer1
        self.encoder2 = self.class_model.layer2
        self.encoder3 = self.class_model.layer3
        self.encoder4 = self.class_model.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x1, x2, x3):

        x1 = self.encoder0(x1)
        e1_1 = self.encoder1(x1)
        x2 = self.encoder0(x2)
        e1_2 = self.encoder1(x2)
        x3 = self.encoder0(x3)
        e1_3 = self.encoder1(x3)

        e1 = e1_1 + e1_2 + e1_3  # maybe concat? (and change num filters)

        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class UNet_class(nn.Module):
    def __init__(self, model, num_classes=1, filters_mult=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        filters = [filters_mult * 64, filters_mult * 128, filters_mult * 256, filters_mult * 512]
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = model

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(filters[3], filters[2], filters[2], is_deconv)

        self.dec5 = DecoderBlock(filters[3] + filters[2], filters[2], filters[2], is_deconv)
        self.dec4 = DecoderBlock(filters[2] + filters[2], filters[2], filters[2], is_deconv)
        self.dec3 = DecoderBlock(filters[1] + filters[2], filters[1], filters[0], is_deconv)
        self.dec2 = DecoderBlock(filters[0] + filters[0], filters[0], filters[0], is_deconv)
        self.dec1 = DecoderBlock(filters[0], filters[0], filters_mult * num_filters, is_deconv)
        self.dec0 = ConvRelu(filters_mult * num_filters, filters_mult * num_filters)
        self.final = nn.Conv2d(filters_mult * num_filters, num_classes, kernel_size=1)
        self.final_sigm = nn.Sequential(nn.Conv2d(filters_mult * num_filters, num_classes, kernel_size=1),
                                        nn.Sigmoid(), )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final_sigm(dec0)

        return x_out


class UNetD_class(nn.Module):
    def __init__(self, model, num_classes=1, filters_mult=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        filters = [filters_mult * 64, filters_mult * 128, filters_mult * 256, filters_mult * 512]
        self.num_classes = num_classes

        self.encoder = model

        self.pool = nn.MaxPool2d(2, 2)

        layer0_modules = [
            ('conv1', nn.Conv2d(3, filters[0] // 2, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(filters[0] // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(filters[0] // 2, filters[0], 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(filters[0])),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(filters[0], filters[0], 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(filters[0])),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(2, 2))  ### maybe? # nn.MaxPool2d(3, stride=2, ceil_mode=True)
        ]

        self.conv1 = nn.Sequential(OrderedDict(layer0_modules))

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(filters[3], filters[2], filters[2], is_deconv)

        self.dec5 = DecoderBlock(filters[3] + filters[2], filters[2], filters[2], is_deconv)
        self.dec4 = DecoderBlock(filters[2] + filters[2], filters[2], filters[2], is_deconv)
        self.dec3 = DecoderBlock(filters[1] + filters[2], filters[1], filters[0], is_deconv)
        self.dec2 = DecoderBlock(filters[0] + filters[0], filters[0], filters[0], is_deconv)
        self.dec1 = DecoderBlock(filters[0], filters[0], filters_mult * num_filters, is_deconv)
        self.dec0 = ConvRelu(filters_mult * num_filters, filters_mult * num_filters)
        self.final = nn.Conv2d(filters_mult * num_filters, num_classes, kernel_size=1)
        self.final_sigm = nn.Sequential(nn.Conv2d(filters_mult * num_filters, num_classes, kernel_size=1),
                                        nn.Sigmoid(), )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final_sigm(dec0)

        return x_out


class SE_class_LinkNet_network_4_3(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = nn.Sequential(*list(model.children()))
        self.num_classes = num_classes

        self.prepare = nn.Sequential(nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU())

        self.encoder0 = self.class_model[:4]
        self.encoder1 = self.class_model[4]
        self.encoder2 = self.class_model[5]
        self.encoder3 = self.class_model[6]
        self.encoder4 = self.class_model[7]

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x):

        x = self.prepare(x)

        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            # f5 = self.finalconv3(f4)
            # x_out = F.log_softmax(f5, dim=1)
            x_out = self.finalconv3(f4)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SE_class_LinkNet_network_4subst3(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = nn.Sequential(*list(model.children()))
        self.num_classes = num_classes

        self.encoder0 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU())

        self.encoder1 = self.class_model[4]
        self.encoder2 = self.class_model[5]
        self.encoder3 = self.class_model[6]
        self.encoder4 = self.class_model[7]

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x):

        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            # f5 = self.finalconv3(f4)
            # x_out = F.log_softmax(f5, dim=1)
            x_out = self.finalconv3(f4)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SE_LinkNet_4_3(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = model
        self.num_classes = num_classes

        self.prepare = nn.Sequential(nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True))

        self.encoder0 = self.class_model.layer0
        self.encoder1 = self.class_model.layer1
        self.encoder2 = self.class_model.layer2
        self.encoder3 = self.class_model.layer3
        self.encoder4 = self.class_model.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x):

        x = self.prepare(x)
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class SE_LinkNet_4subst3(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        self.class_model = model
        self.num_classes = num_classes

        # if input_3x3:
        #     layer0_modules = [
        #         ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
        #                             bias=False)),
        #         ('bn1', nn.BatchNorm2d(64)),
        #         ('relu1', nn.ReLU(inplace=True)),
        #         ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
        #                             bias=False)),
        #         ('bn2', nn.BatchNorm2d(64)),
        #         ('relu2', nn.ReLU(inplace=True)),
        #         ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
        #                             bias=False)),
        #         ('bn3', nn.BatchNorm2d(64)),
        #         ('relu3', nn.ReLU(inplace=True)),
        #     ]
        # else:
        layer0_modules = [
            ('conv1', nn.Conv2d(4, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.class_model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.encoder0 = self.class_model.layer0
        self.encoder1 = self.class_model.layer1
        self.encoder2 = self.class_model.layer2
        self.encoder3 = self.class_model.layer3
        self.encoder4 = self.class_model.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.final = nn.Sequential(nn.Conv2d(32, num_classes, 2, padding=1),
                                   nn.Sigmoid(), )

    def forward(self, x):

        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Summ Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            f5 = self.finalconv3(f4)
            x_out = F.log_softmax(f5, dim=1)
        else:
            f5 = self.final(f4)
            x_out = f5
        return x_out


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch * 2),
                           nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UneXt50(nn.Module):
    def __init__(self, num_classes=5, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32 + 16 * 4, num_classes, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.log_softmax(x, dim=1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


class UneXt50_mltlbl(nn.Module):
    def __init__(self, num_classes=5, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32 + 16 * 4, num_classes, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


class UneXt50_mlthead(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x1 = self.final_conv(self.drop(x))
        x1 = F.sigmoid(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x2 = self.final_conv(self.drop(x))
        x2 = F.sigmoid(x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.final_conv(self.drop(x))
        x3 = F.sigmoid(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        return x1, x2, x3
