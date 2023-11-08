import matplotlib.pyplot as plt
import numpy as np
import pickle
from loguru import logger
import arguments
import torchvision
import torch
import os
from torch.utils.data import Dataset
import PIL.Image as Image


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

def set_idx(imidx, imidx_list, idx_shuffle):

    '''choose set image or random'''
    args = arguments.Arguments(logger)
    if args.get_imidx() == 000000:
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
    else:
        idx = args.get_imidx()
        imidx_list.append(idx)

    return  idx, imidx_list

def label_mapping(origin_label):
    args = arguments.Arguments(logger)
    if args.get_dataset() == 'mnist':
        if origin_label < 5:
            tmp_label_1 = torch.Tensor([0]).long()
        else:
            tmp_label_1 = torch.Tensor([1]).long()
    elif args.get_dataset() == 'cifar100':
        mapping_dict = {0: [4, 30, 55, 72, 95],
                        1: [1, 32, 67, 73, 91],
                        2: [54, 62, 70, 82, 92],
                        3: [9, 10, 16, 28, 61],
                        4: [0, 51, 53, 57, 83],
                        5: [22, 39, 40, 86, 87],
                        6: [5, 20, 25, 84, 94],
                        7: [6, 7, 14, 18, 24],
                        8: [3, 42, 43, 88, 97],
                        9: [12, 17, 37, 68, 76],
                        10: [23, 33, 49, 60, 71],
                        11: [15, 19, 21, 31, 38],
                        12: [34, 63, 64, 66, 75],
                        13: [26, 45, 77, 79, 99],
                        14: [2, 11, 35, 46, 98],
                        15: [27, 29, 44, 78, 93],
                        16: [36, 50, 65, 74, 80],
                        17: [47, 52, 56, 59, 96],
                        18: [8, 13, 48, 58, 90],
                        19: [41, 69, 81, 85, 89]}
        tmp_label_1 = torch.Tensor([k for k, v in mapping_dict.items() if origin_label in v]).long()
    else:
        tmp_label_1 = torch.Tensor([origin_label]).long()

    return tmp_label_1


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


# def load_saved_data_loader(file_obj):
#     return pickle.load(file_obj)
#
# def load_data_loader_from_file(logger, filename):
#     """
#     Loads DataLoader object from a file if available.
#
#     :param logger: loguru.Logger
#     :param filename: string
#     """
#     logger.info("Loading data loader from file: {}".format(filename))
#
#     with open(filename, "rb") as f:
#         return load_saved_data_loader(f)
#
# def load_train_data_loader(logger, args):
#     """
#     Loads the training data DataLoader object from a file if available.
#
#     :param logger: loguru.Logger
#     :param args: Arguments
#     """
#     if os.path.exists(args.get_train_data_loader_pickle_path()):
#         return load_data_loader_from_file(logger, args.get_train_data_loader_pickle_path())
#     else:
#         logger.error("Couldn't find train data loader stored in file")
#
#         raise FileNotFoundError("Couldn't find train data loader stored in file")
#
# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# args = arguments.Arguments(logger)
#
# # get some random training images
# trainloader = load_train_data_loader(logger = logger, args = args)
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# args.get_logger().info("The label of image #{}", ' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size)))