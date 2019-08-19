# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/7/29 0029, matt '
import  argparse


def get_augments():
    parser = argparse.ArgumentParser(description="pytorch cifar test")
    parser.add_argument("--conf", type=str, default="config", help="load config file")
    parser.add_argument("--net", type=str, default="VGG13", choices=("VGG11", "VGG13", "VGG16", "VGG19", "GoogLeNet",
                                                                     "ResNet18", "ResNet34", "ResNet50", "ResNet101",
                                                                     "ResNet152", "DRN_C_26", "DRN_C_42", "DRN_C_58",
                                                                     "DRN_D_54", "DRN_A_50", "DenseNet_cifar"),
                        help="net select")
    parser.add_argument("--action", type=str, default="train", choices=("train", ))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pre_train", type=int, default=0)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)

    return parser.parse_args()


def main():
    args = get_augments()
    conf = __import__(args.conf, globals(), locals(), ["Config"]).Config()
    if args.action == "train":
        import train
        train.run(conf, args.net, args.lr, args.decay, args.epochs, pre_train=args.pre_train)


if __name__ == "__main__":
    main()
