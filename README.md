# Train CIFAR10 with pytorch
The main purpose of this project is to familiarize various commomly 
used convolutional network, such as VGG, GoogLeNet, ResNet, etc.

The code is mainly reference https://github.com/kuangliu/pytorch-cifar

## Prerequisites
* Python 3.7+
* Pytorch 1.0+
* visdom 

## Usage
You can use follow command to train

    python main.py --net="VGG16" --lr=0.01 --decay=0.001 --epochs=10 --pre_train=0