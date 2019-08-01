# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '

from model.vgg import VGG
from model.googlenet import GoogLeNet
from model.resnet import get_res_net
from model.drn import get_drn_net

__all__ = ["VGG", "GoogLeNet", "get_res_net", "get_drn_net"]
