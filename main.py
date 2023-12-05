# from utils import data_download
#
# if __name__ =="__main__":
#     data_download

from loguru import logger
import logging
import arguments
import time
import os
import time
from plot import plot
from os import listdir
import numpy as np
from utils.methods import dlg, idlg, dlgadam, invg, mdlg, cpa, mdlg_mt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from ica import ica1
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
from utils import files
# logging.getLogger().setLevel(Logging.INFO)
# from lpips_pytorch import LPIPS, lpips
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
from utils.net import LeNet, FC2
from utils.data_download import load_data
from utils.net_utils import intialize_nets
from utils.save import save_results


def main():
    args = arguments.Arguments(logger)

    # Initialize logger
    log_files, str_time = files.files(args)
    handler = logger.add(log_files[0], enqueue=True)

    dataset = args.get_dataset()
    root_path = args.get_root_path()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, args.get_debugOrRun()+'/compare_%s' % dataset).replace('\\', '/')
    # eval_res_path = os.path.join(root_path, '/compare_%s' % dataset).replace('\\', '/')
    lr = args.get_lr()
    num_dummy = args.get_num_dummy()
    Iteration = args.get_iteration()
    num_exp = args.get_num_exp()
    methods = args.get_methods()
    log_interval = args.get_log_interval()
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    args.log()
    args.logger.info('dataset is #{}:', dataset)
    args.logger.info('lr is #{}', lr)
    args.logger.info('log interval is #{}', log_interval)
    args.logger.info('root path is #{}:', root_path)
    args.logger.info('data_path is #{}:', data_path)
    args.logger.info('save_path is #{}:', save_path)

    tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle, mean_std = load_data(dataset = dataset, root_path = root_path, data_path = data_path, save_path = save_path)

    ''' train DLG and iDLG and mDLG and DLGAdam'''
    for idx_net in range(num_exp):

        args.logger.info('running #{}|#{} experiment', idx_net, num_exp)
        imidx_lists = []
        final_iters = []
        final_imgs = []

        '''train on different methods'''
        for method in methods: #
            args.logger.info('#{}, Try to generate #{} images', method, num_dummy)
            if method == 'DLG':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img, results = dlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')

            elif method == 'iDLG':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img, results  = idlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')

            elif method == 'DLGAdam':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img, results = dlgadam(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')


            elif method == 'InvG':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)

                imidx_list, final_iter, final_img, results  = invg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')


            elif method == 'mDLG':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img ,results = mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')


            elif method == 'mDLG_mt':
                nets = intialize_nets(method = method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                            alter_num_classes=alter_num_classes, input_size=input_size)
                for i in range(len(nets)):
                    nets[i] = nets[i].to(device)
                    args.logger.info('Size of net #{} is #{}',i, len(nets[i].state_dict()))
                imidx_list, final_iter, final_img, results = mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, nets, num_classes, Iteration, save_path, str_time)
                save_results(results, root_path + '/' + method + '_' + str(imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(args.num_servers) + '_' + str_time + '.csv')

            elif method == 'CPA':
                imidx_list, final_iter, final_img = cpa(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path, str_time)
                imidx_lists.append(imidx_list)
                final_iters.append(final_iter)
                final_imgs.append(final_img)
                plt.imshow(final_img[0][0])
                plt.title('%s_on_iter=%d' % (method, final_iter))
                plt.axis('off')
                plt.savefig('%s/CPA_final_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[0]))
                plt.close()

            # width, height = final_imgs[0][0][0].size
            # res = Image.new(final_imgs[0][0][0].mode, (width , height * len(final_imgs)))
            # for i, im in enumerate(final_imgs[0][0]):
            #     plt.imshow(im)
            #     res.paste(im, box=(0, i * height))
            #
            # res.save('compare_all_cifar100_test.png')
            # result = plot.plot(mode = 'h', input_filenames=final_imgs)
            # result.save('compare_all_cifar100_test.png')

            # plt.imshow(final_img[0][0])
            # plt.title('iter=%d' % (final_iter))
            # plt.axis('off')
            # plt.savefig('%s/DLG_final_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[0]))
            # plt.close()
        # save_results(results, root_path + '/' + method + '_' + str(imidx_list[idx_net]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(int(time.time())) + '.csv')
        print(imidx_list)
        args.logger.info('imidx_list: #{}', imidx_list)


        # args.logger.info('gt_label: #{}', gt_label.detach().cpu().data.numpy())
        #
        # for method in methods:
        #     args.logger.info('method: #{}, "loss: #{}, mse: #{}, label: #{}', method, str(loss[-1]), str(mse[-1]), str(label))
        # print('----------------------\n\n')
    logger.remove(handler)


if __name__ == '__main__':
    main()

