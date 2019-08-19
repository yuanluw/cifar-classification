# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from datetime import datetime
import visdom
import numpy as np
import os

from utils import *
from dataset import *


viz = visdom.Visdom(env="model_test")


def train(net, train_data, test_data, num_epochs, optimizer, criterion):
    net = net.cuda()
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    test_loss_list = []
    prev_time = datetime.now()
    print("start training: ", prev_time)
    train_acc_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="train_acc"))
    train_loss_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="train_loss"))
    test_acc_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="test_acc"))
    test_loss_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title="test_loss"))
    train_y_axis = 0
    test_y_axis = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, verbose=True)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0

        net = net.train()
        i = 0
        for im, label in train_data:
            im = Variable(im.cuda())
            label = Variable(label.cuda())
            output = net(im)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            cur_time = datetime.now()
            time_str = count_time(prev_time, cur_time)
            train_acc += get_acc(output, label)
            train_acc_list.append(get_acc(output, label))
            train_loss_list.append(loss.item())
            viz.line(X=np.array([train_y_axis]), Y=np.array([get_acc(output, label)]), win=train_acc_win,
                     update="append")
            viz.line(X=np.array([train_y_axis]), Y=np.array([loss.item()]), win=train_loss_win, update="append")
            # print("train: current %d batch loss is %f acc is %f time is %s"%(i, loss.item, train_acc, time_str))
            train_y_axis += 1
            i += 1
        if test_data is not None:
            net = net.eval()
            test_loss = 0.0
            test_acc = 0
            j = 0
            for im, label in test_data:
                with torch.no_grad():
                    im = Variable(im.cuda())
                    label = Variable(label.cuda())
                output = net(im)
                loss = criterion(output, label)
                test_loss += loss.item()
                cur_time = datetime.now()
                time_str = count_time(prev_time, cur_time)
                test_acc += get_acc(output, label)
                test_acc_list.append(get_acc(output, label))
                test_loss_list.append(loss.item())
                viz.line(X=np.array([test_y_axis]), Y=np.array([get_acc(output, label)]), win=test_acc_win,
                         update="append")
                viz.line(X=np.array([test_y_axis]), Y=np.array([loss.item()]), win=test_loss_win, update="append")
                # print("test: current %d batch loss is %f acc is %f time is %s"%(i, loss.item, train_acc, time_str))
                j += 1
                test_y_axis += 1
            epoch_str = ("epoch %d, train loss: %f train acc: %f test_loss: %f test_acc: %f" % (
                epoch, train_loss / len(train_data), train_acc / len(train_data), test_loss / len(test_data),
                test_acc / len(test_data)
            ))
        else:
            epoch_str = ("epoch %d, trian loss: %f, train acc: %f" % (epoch, train_loss/len(train_data), train_acc/len(
                train_data)))

        print(epoch_str)
        scheduler.step(train_acc/len(train_data))
    print("end time: ", datetime.now())
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def run(conf, net_name,learning_rate, decay_rate, num_epoch, pre_train=False):
    print("learning_rate: %f epoch_num: %d decay_rate: %f pre_train is %d" % (learning_rate, num_epoch, decay_rate,
                                                                              pre_train))
    train_data = get_dataset(is_train=True, num_worker=conf.num_worker, batch_size=conf.train_batch_size)
    test_data = get_dataset(is_train=False, num_worker=conf.num_worker, batch_size=conf.test_batch_size)

    net = conf.get_net(net_name)

    if conf.Mul_GPU:
        net = nn.DataParallel(net)
    if pre_train:
        net.load_state_dict(torch.load(os.path.join("pre_train", str(net_name + "_.pkl"))))
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)
    criterion = nn.CrossEntropyLoss()

    train_loss_list, train_acc_list, test_loss_list, test_acc_list = train(net, train_data, test_data, num_epoch,
                                                                           optimizer, criterion)
    if pre_train:
        os.remove(os.path.join("pre_train", str(net_name + "_.pkl")))
    torch.save(net.state_dict(), os.path.join("pre_train", str(net_name + "_.pkl")))

