# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '

from model import *


class Config(object):
    train_batch_size = 128
    test_batch_size = 64
    num_worker = 4
    input_image_w = 32
    input_image_h = 32
    Mul_GPU = True

    def get_net(self, net_name):
        if net_name.find("VGG") >= 0:
            return VGG(net_name)
        elif net_name == "GoogLeNet":
            return GoogLeNet()
        elif net_name.find("ResNet") >= 0:
            return get_res_net(net_name)
        elif net_name.find("DRN") >= 0:
            return get_drn_net(net_name)



