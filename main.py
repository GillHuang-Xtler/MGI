# from utils import data_download
#
# if __name__ =="__main__":
#     data_download

from loguru import logger
import logging
import arguments
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
from utils import files
# logging.getLogger().setLevel(Logging.INFO)
# from lpips_pytorch import LPIPS, lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def load_data(dataset, root_path, data_path, save_path):


    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])



    if not os.path.exists('res'):
        os.mkdir('res')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=False)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)


    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    return tt, tp, num_classes, channel, hidden, dst

def main():
    args = arguments.Arguments(logger)

    # Initialize logger
    log_files = files.files(args.get_start_index_str(), args.get_num_exp())
    handler = logger.add(log_files[0], enqueue=True)

    dataset = args.get_dataset()
    root_path = args.get_root_path()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/compare_%s' % dataset).replace('\\', '/')
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
    args.logger.info('root path is #{}:', root_path)
    args.logger.info('data_path is #{}:', data_path)
    args.logger.info('save_path is #{}:', save_path)

    tt, tp, num_classes, channel, hidden, dst = load_data(dataset = dataset, root_path = root_path, data_path = data_path, save_path = save_path)

    ''' train DLG and iDLG and mDLG and DLGAdam'''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        net_1 = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net_1.apply(weights_init)

        args.logger.info('running #{}|#{} experiment', idx_net, num_exp)
        net = net.to(device)
        net_1 = net_1.to(device)

        idx_shuffle = np.random.permutation(len(dst))

        for method in methods: #
            args.logger.info('#{}, Try to generate #{} images', method, num_dummy)

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            if method == 'mDLG':

                # compute original gradient
                out = net(gt_data)
                out_1 = net_1(gt_data)
                y = criterion(out, gt_label)
                y_1 = criterion(out_1, gt_label)
                dy_dx = torch.autograd.grad(y, net.parameters())
                dy_dx_1 = torch.autograd.grad(y_1, net_1.parameters())
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))
                original_dy_dx_1 = list((_.detach().clone() for _ in dy_dx_1))

            elif method == "DLG" or method == "iDLG" or method == 'DLGAdam' or method == 'InvG' or method == 'CPA' :
                # compute original gradient
                out = net(gt_data)
                y = criterion(out, gt_label)
                dy_dx = torch.autograd.grad(y, net.parameters())
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            else:
                print("unknown methods")
                continue

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)
            elif method == 'DLGAdam':
                lr = 0.1
                optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
            elif method == 'InvG':
                lr = 0.1
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
            elif method == 'CPA':
                lr = 0.01
                optimizer = torch.optim.Adam([dummy_data, dummy_label], lr = 0.01)
            elif method =="mDLG":
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)
                label_pred_1 = torch.argmin(torch.sum(original_dy_dx_1[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)
            else:
                continue

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            args.logger.info('lr = #{}', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    pred_1 = net_1(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)
                    elif method == 'InvG':
                        dummy_loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    elif method == 'DLGAdam':
                        dummy_loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    elif method == 'CPA':
                        dummy_loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    elif method == "mDLG":
                        dummy_loss = criterion(pred, label_pred)
                        dummy_loss_1 = criterion(pred_1, label_pred_1)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    if method == "mDLG":
                        dummy_dy_dx_1 = torch.autograd.grad(dummy_loss_1, net_1.parameters(), create_graph=True)

                    grad_diff = 0

                    if method =='iDLG' or method == "DLG" or method == 'DLGAdam' :
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                    elif method == 'InvG':
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            cos = nn.CosineSimilarity(dim = 0)
                            grad_diff += (1-cos(gx, gy)).sum()
                        grad_diff.backward()
                    elif method == 'CPA':
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += (lpips(gx, gy)).sum()
                    elif method == "mDLG":
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        for gx_1, gy_1 in zip(dummy_dy_dx_1, original_dy_dx_1):
                            grad_diff += ((gx_1 - gy_1) ** 2).sum()
                        grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if iters % int(Iteration / log_interval) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    args.logger.info( 'current time: #{}, current iter #{}, loss = #{}, mse = #{}', current_time, iters, current_loss, mses[-1])
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'DLGAdam':
                            plt.savefig('%s/DLGAdam_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'InvG':
                            plt.savefig('%s/InvG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'CPA':
                            plt.savefig('%s/CPA_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'mDLG':
                            plt.savefig('%s/mDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        else:
                            continue

                    if abs(current_loss) < 0.000001:  # converge
                        break

            if method == 'DLG':
                loss= losses
                label = torch.argmax(dummy_label, dim=-1).detach().item()
                mse = mses
                args.logger.info('loss of DLG: #{}, mse of DLG: #{}, label of DLG: #{}', loss[-1], mse[-1],  label)  #
            elif method == 'iDLG':
                loss = losses
                label = label_pred.item()
                mse = mses
                args.logger.info('loss of iDLG: #{}, mse of iDLG: #{}, label of iDLG: #{}', loss[-1], mse[-1],  label)  #
            elif method == 'DLGAdam':
                loss= losses
                label = torch.argmax(dummy_label, dim=-1).detach().item()
                mse = mses
                args.logger.info('loss of DLGAdam: #{}, mse of DLGAdam: #{}, label of DLGAdam: #{}', loss[-1], mse[-1],  label)  #
            elif method == 'InvG':
                loss = losses
                label = torch.argmax(dummy_label, dim=-1).detach().item()
                mse = mses
                args.logger.info('loss of InvG: #{}, mse of InvG: #{}, label of InvG: #{}', loss[-1], mse[-1],  label)  #
            elif method == 'CPA':
                loss = losses
                label = label_pred.item()
                mse = mses
                args.logger.info('loss of CPA: #{}, mse of CPA: #{}, label of CPA: #{}', loss[-1], mse[-1],  label)  #
            elif method == 'mDLG':
                loss = losses
                label = label_pred.item()
                mse = mses
                args.logger.info('loss of mDLG: #{}, mse of mDLG: #{}, label of mDLG: #{}', loss[-1], mse[-1],  label)  #

            else:
                continue

        args.logger.info('imidx_list: #{}', imidx_list)
        # args.logger.info('gt_label: #{}', gt_label.detach().cpu().data.numpy())
        #
        # for method in methods:
        #     args.logger.info('method: #{}, "loss: #{}, mse: #{}, label: #{}', method, str(loss[-1]), str(mse[-1]), str(label))
        # print('----------------------\n\n')
    logger.remove(handler)


if __name__ == '__main__':
    main()

