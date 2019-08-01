# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/31 0031, matt '

# http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf

import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):
    def __init__(self, block, layers, num_classes=10, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False,
                 out_middle=False, pool_size=4, arch='D', verbose=False):
        super(DRN, self).__init__()
        self.in_planes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.verbose = verbose
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)

        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], dilation=4,
                                                                   new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6],
                                                                       dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7],
                                                                       dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = list()
        layers.append(block(
            self.in_planes, planes, stride, downsample, dilation=(1, 1) if dilation == 1 else(
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, residual=residual, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.in_planes, channels, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
            self.in_planes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()
        if self.arch == 'C':
            x = self.conv1(x)
            if self.verbose:
                print("conv1 size: ", x.size())
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)
            if self.verbose:
                print("layer0 size: ", x.size())

        x = self.layer1(x)
        if self.verbose:
            print("layer1 size: ", x.size())
        y.append(x)
        x = self.layer2(x)
        if self.verbose:
            print("layer2 size: ", x.size())
        y.append(x)
        x = self.layer3(x)
        if self.verbose:
            print("layer3 size: ", x.size())
        y.append(x)
        x = self.layer4(x)
        if self.verbose:
            print("layer4 size: ", x.size())
        y.append(x)
        x = self.layer5(x)
        if self.verbose:
            print("layer5 size: ", x.size())
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            if self.verbose:
                print("layer6 size: ", x.size())
            y.append(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            if self.verbose:
                print("layer7 size: ", x.size())
            y.append(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            if self.verbose:
                print("layer8 size: ", x.size())
            y.append(x)
        if self.out_map:
            x = self.fc(x)
            if self.verbose:
                print("fc size: ", x.size())
        else:
            x = self.avgpool(x)
            if self.verbose:
                print("avgpool size: ", x.size())
            x = self.fc(x)
            if self.verbose:
                print("fc size: ", x.size())
            x = x.view(x.size(0), -1)
            if self.verbose:
                print("view size: ", x.size())

        if self.out_middle:
            return x, y
        else:
            return x


class DRN_A(nn.Module):
    def __init__(self, block, layers, num_classes=10, verbose=False):
        self.in_planes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.verbose = verbose
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.verbose:
            print("conv1 size: ", x.size())
        x = self.maxpool(x)
        if self.verbose:
            print("maxpool size: ", x.size())
        x = self.layer1(x)
        if self.verbose:
            print("layer1 size: ", x.size())
        x = self.layer2(x)
        if self.verbose:
            print("layer2 size: ", x.size())
        x = self.layer3(x)
        if self.verbose:
            print("layer3 size: ", x.size())
        x = self.layer4(x)
        if self.verbose:
            print("layer4 size: ", x.size())

        x = self.avgpool(x)
        if self.verbose:
            print("avgpool size: ", x.size())
        x = x.view(x.size(0), -1)
        if self.verbose:
            print("view size: ", x.size())
        x = self.fc(x)
        if self.verbose:
            print("fc size: ", x.size())
        return x


def drn_a_50(verbose=False):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], verbose=verbose)
    return model


def drn_c_26(verbose=False):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', verbose=verbose)
    return model


def drn_c_42(verbose=False):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', verbose=verbose)
    return model


def drn_c_58(verbose=False):
    return DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', verbose=verbose)


def drn_d_54(verbose=False):
    return DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', verbose=verbose)


def get_drn_net(net_name):
    if net_name == "DRN_C_26":
        return drn_c_26()
    elif net_name == "DRN_C_42":
        return drn_c_42()
    elif net_name == "DRN_C_58":
        return drn_c_58()
    elif net_name == "DRN_D_54":
        return drn_d_54()
    elif net_name == "DRN_A_50":
        return drn_a_50()


if __name__ == "__main__":
    net = drn_d_54(verbose=True)
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y = net(x)
    print(y.size())
