# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '

# https://arxiv.org/pdf/1409.1556.pdf

import torch
import torch.nn as nn
from torchvision import models

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, verbose=False):
        super(VGG, self).__init__()
        self.verbose = verbose
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        if self.verbose:
            print("features out:", out.size())
        out = out.view(out.size(0), -1)
        if self.verbose:
            print("flatten out:", out.size())
        out = self.classifier(out)
        if self.verbose:
            print("classifier out:", out.size())
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        print(cfg)
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = models.vgg16(pretrained=True)
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())