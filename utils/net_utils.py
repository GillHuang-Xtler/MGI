from utils import FC2, LeNet, MNISTCNN, Cifar100ResNet
from utils.data_processing import weights_init

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
                net = Cifar100ResNet(num_classes = num_classes)
                # net.apply(weights_init)
                nets.append(net)
        elif args.get_net == 'resnet':
            for i in range(num_servers):
                net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
                net.apply(weights_init)
                nets.append(net)
        return nets

    else:
        args.logger.info('running simgle server')
        nets = []
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
        return nets

