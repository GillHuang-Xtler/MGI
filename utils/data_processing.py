import matplotlib.pyplot as plt
import numpy as np
import pickle
from loguru import logger
import arguments
import torchvision
import os
from torch.utils.data import Dataset


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