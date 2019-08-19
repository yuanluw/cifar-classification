# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/19 0019, matt '


import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, verbose=False):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        self.verbose = verbose

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.verbose:
            print("conv1 size: ", out.size())
        out = self.trans1(self.dense1(out))
        if self.verbose:
            print("dense1 size: ", out.size())
        out = self.trans2(self.dense2(out))
        if self.verbose:
            print("dense2 size: ", out.size())
        out = self.trans3(self.dense3(out))
        if self.verbose:
            print("dense3 size: ", out.size())
        out = self.dense4(out)
        if self.verbose:
            print("dense4 size: ", out.size())
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        if self.verbose:
            print("avg_pool2d size: ", out.size())
        out = out.view(out.size(0), -1)
        if self.verbose:
            print("view size: ", out.size())
        out = self.linear(out)
        if self.verbose:
            print("linear size: ", out.size())
        return out


def DenseNet121(verbose=False):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, verbose=verbose)


def DenseNet169(verbose=False):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, verbose=verbose)


def DenseNet201(verbose=False):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, verbose=verbose)


def DenseNet161(verbose=False):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, verbose=verbose)


def DenseNet_cifar(verbose=False):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, verbose=verbose)


def get_dense_net(net_name):
    if net_name == "DenseNet121":
        return DenseNet121()
    elif net_name == "DenseNet169":
        return DenseNet169()
    elif net_name == "DenseNet201":
        return DenseNet201()
    elif net_name == "DenseNet161":
        return DenseNet161()
    elif net_name == "DenseNet_cifar":
        return DenseNet_cifar()


if __name__ == "__main__":
    x = torch.randn((1, 3, 32, 32))
    net = DenseNet_cifar(verbose=True)
    print(net)

    y = net(x)
