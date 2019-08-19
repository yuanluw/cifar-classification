# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataset(is_train=True, num_worker=2, batch_size=64):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if is_train:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        return train_loader
    else:
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        return test_loader


if __name__ == "__main__":
    train_set = torchvision.datasets.VOCSegmentation("./data", year='2012', image_set='train', download=True)
    val_set = torchvision.datasets.VOCSegmentation("./data", year='2012', image_set='val', download=True)
    test_set = torchvision.datasets.VOCSegmentation('./data', year='2012', image_set='trainval', download=True)
