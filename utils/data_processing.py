import matplotlib.pyplot as plt
import numpy as np
import pickle
from loguru import logger
import arguments
import torchvision
import os

def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)

def load_data_loader_from_file(logger, filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    logger.info("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)

def load_train_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_train_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_train_data_loader_pickle_path())
    else:
        logger.error("Couldn't find train data loader stored in file")

        raise FileNotFoundError("Couldn't find train data loader stored in file")

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

args = arguments.Arguments(logger)

# get some random training images
trainloader = load_train_data_loader(logger = logger, args = args)
dataiter = iter(trainloader)
images, labels = next(dataiter)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
args.get_logger().info("The label of image #{}", ' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size)))