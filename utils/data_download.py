import torch
import torchvision
from arguments import Arguments
from loguru import logger
import os
import pathlib
import pickle
from torchvision import datasets, transforms
import numpy as np
from .data_processing import Dataset_from_Image

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
    if dataset == 'mnist':
        shape_img = (28, 28)
        input_size = 28
        num_classes = 10
        alter_num_classes = 2
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        input_size = 32
        num_classes = 100
        alter_num_classes = 20
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)


    elif dataset == 'stl10':
        shape_img = (96,96)
        input_size = 96
        num_classes = 10
        alter_num_classes = 2
        channel = 3
        hidden = 6912
        dst = datasets.STL10(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        input_size = 32
        num_classes = 5749
        alter_num_classes = 2
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, './data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)
        # dst = torchvision.datasets.LFWPeople(data_path, download = True)

    else:
        exit('unknown dataset')

    idx_shuffle = np.random.permutation(len(dst))

    return tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle

# def save_data_loader_to_file(data_loader, file_obj):
#     pickle.dump(data_loader, file_obj)
#
# args = Arguments(logger)
#
# if 'Cifar10' in args.dataset:
#     # ---------------------------------
#     # ------------ CIFAR10 ------------
#     # ---------------------------------
#
#     args.get_logger().info("........Start to download Cifar10 Dataset.........")
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     batch_size = args.batch_size
#
#     trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
#                                             download=True, transform=transform)
#     train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                               shuffle=True, num_workers=2)
#
#     testset = torchvision.datasets.CIFAR10(root='../data', train=False,
#                                            download=True, transform=transform)
#     test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                              shuffle=False, num_workers=2)
#     args.get_logger().info("Training set size #{}", str(len(trainset)))
#     args.get_logger().info("Testing set size #{}", str(len(testset)))
#
#
#     TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
#     TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"
#     if not os.path.exists("data_loaders/cifar10"):
#         pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)
#
#     with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
#         save_data_loader_to_file(train_data_loader, f)
#
#     with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
#         save_data_loader_to_file(test_data_loader, f)