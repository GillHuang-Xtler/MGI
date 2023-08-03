import torch
import torchvision
import torchvision.transforms as transforms
from arguments import Arguments
from loguru import logger
import os
import pathlib
import pickle


def save_data_loader_to_file(data_loader, file_obj):
    pickle.dump(data_loader, file_obj)

args = Arguments(logger)

if 'Cifar10' in args.dataset:
    # ---------------------------------
    # ------------ CIFAR10 ------------
    # ---------------------------------

    args.get_logger().info("........Start to download Cifar10 Dataset.........")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.batch_size

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    args.get_logger().info("Training set size #{}", str(len(trainset)))
    args.get_logger().info("Testing set size #{}", str(len(testset)))


    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"
    if not os.path.exists("data_loaders/cifar10"):
        pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)