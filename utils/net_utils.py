from utils import FC2, LeNet
from utils.data_processing import weights_init

def intialize_nets(args, method, channel, hidden, num_classes,alter_num_classes, input_size):

    if method == 'mDLG_mt':
        if args.get_net() == 'lenet':
            net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
            net_1 = LeNet(channel=channel, hidden=hidden, num_classes=alter_num_classes)
            net.apply(weights_init)
            net_1.apply(weights_init)
        elif args.get_net() == 'fc2':
            net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
            net_1 = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=alter_num_classes)
    else:
        if args.get_net() == 'lenet':
            net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
            net_1 = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
            net.apply(weights_init)
            net_1.apply(weights_init)
        elif args.get_net() == 'fc2':
            net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
            net_1 = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
        # net.apply(weights_init)
        # net_1.apply(weights_init)
    return net, net_1

