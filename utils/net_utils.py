from utils import FC2, LeNet, MNISTCNN, Cifar100ResNet
from utils.data_processing import weights_init
import torch.nn.init as init
import torch.nn as nn

#
# def intialize_nets(args, method, channel, hidden, num_classes,alter_num_classes, input_size):
#
#     if method == 'mDLG_mt':
#         print('running multi task')
#         if args.get_net() == 'lenet':
#             net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
#             if args.get_dataset() == 'mnist' and args.get_net_mt_diff() == True:
#                 print("different structure")
#                 net_1 = MNISTCNN(channel=channel, hidden=hidden, num_classes=alter_num_classes)
#             elif args.get_dataset() == 'cifar100' and args.get_net_mt_diff() == True:
#                 print("different structure")
#                 net_1 = Cifar100ResNet(num_classes = alter_num_classes)
#             else:
#                 net_1 = LeNet(channel=channel, hidden=hidden, num_classes=alter_num_classes)
#             net.apply(weights_init)
#             net_1.apply(weights_init)
#
#         elif args.get_net() == 'fc2':
#             net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
#             net_1 = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=alter_num_classes)
#
#     elif method == 'mDLG':
#         args.logger.info('running same task multi server')
#         nets = []
#         if args.get_net() == 'lenet':
#             net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
#             net_1 = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
#             net.apply(weights_init)
#             net_1.apply(weights_init)
#         elif args.get_net() == 'fc2':
#             net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
#             net_1 = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
#         # net.apply(weights_init)
#         # net_1.apply(weights_init)
#         elif args.get_net() == 'resnet':
#             net = Cifar100ResNet(num_classes = num_classes)
#             net_1 = Cifar100ResNet(num_classes = num_classes)
#             # init_params(net)
#             # init_params(net_1)
#         if args.get_net() == "lenet":
#             for i in range(args.num_servers):
#                 net = LeNet(num_classes = num_classes)
#                 # net.apply(weights_init)
#                 nets.append(net)
#         nets.append(net)
#         nets.append(net_1)
#         return nets
#
#
#     else:
#         args.logger.info('running simgle server')
#         nets = []
#         if args.get_net() == 'lenet':
#             net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
#             net.apply(weights_init)
#         elif args.get_net() == 'fc2':
#             net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
#         # net.apply(weights_init)
#         elif args.get_net() == 'resnet':
#             net = Cifar100ResNet(num_classes = num_classes)
#             # init_params(net)
#         nets.append(net)
#         return nets

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def intialize_nets(args, method, channel, hidden, num_classes,alter_num_classes, input_size):

    if method == 'mDLG_mt':
        args.logger.info('running different task multi server')
        if args.get_net() == 'lenet':
            net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
            if args.get_dataset() == 'mnist' and args.get_net_mt_diff() == True:
                print("different structure")
                net_1 = MNISTCNN(channel=channel, hidden=hidden, num_classes=alter_num_classes)
            elif args.get_dataset() == 'cifar100' and args.get_net_mt_diff() == True:
                print("different structure")
                net_1 = Cifar100ResNet(num_classes = alter_num_classes)
            else:
                net_1 = LeNet(channel=channel, hidden=hidden, num_classes=alter_num_classes)
            net.apply(weights_init)
            net_1.apply(weights_init)

        elif args.get_net() == 'fc2':
            net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
            net_1 = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=alter_num_classes)

    elif method == 'mDLG':
        args.logger.info('running same task multi server')
        num_servers = args.num_servers
        args.logger.info('number of servers: #{}', num_servers)
        nets = []
        if args.get_net() == "lenet":
            for i in range(num_servers):
                net = LeNet(num_classes = num_classes)
                net.apply(weights_init)
                nets.append(net)
        elif args.get_net() == 'resnet':
            for i in range(num_servers):
                net = Cifar100ResNet(num_classes=num_classes)
                init_params(net)
                nets.append(net)
        elif args.get_net() == 'fc2':
            for i in range(num_servers):
                net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
                net.apply(weights_init)
                nets.append(net)
        return nets

    else:
        args.logger.info('running simgle server')
        nets = []
        for i in range(args.num_servers):
            if args.get_net() == 'lenet':
                net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
                net.apply(weights_init)
                nets.append(net)
            elif args.get_net() == 'fc2':
                net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
            # net.apply(weights_init)
                nets.append(net)
            elif args.get_net() == 'resnet':
                net = Cifar100ResNet(num_classes = num_classes)
                net.apply(weights_init)
                nets.append(net)
        print("num nets" + str(len(nets)))
        return nets

