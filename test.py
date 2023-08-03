import matplotlib.pyplot as plt
import numpy as np
import pickle
from loguru import logger
import arguments
import torchvision
import os
import torch
import random
import PIL

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
    # (3, 32, 32)
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

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.99)

def init_nets_and_save():
    net_0 = Net()
    net_0_full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10FC0.model")
    torch.save(net_0.state_dict(), net_0_full_save_path)

    # change initialization
    net_1 = Net()
    net_1.apply(init_weights)
    net_1_full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10FC1.model")
    torch.save(net_1.state_dict(), net_1_full_save_path)

# train the model for 1 round.

def train_model(net_0_full_save_path, net_1_full_save_path):

    # load nets
    model_0 = Net()
    model_0.load_state_dict(torch.load(net_0_full_save_path))
    model_1 = Net()
    model_1.load_state_dict(torch.load(net_1_full_save_path))

    # check if nets are the same
    for p1, p2 in zip(model_0.parameters(), model_1.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print("different")
        else:
            print("same")

    # train nets using the same single data
    import torch.optim as optim

    model_id = 0
    for net in [model_0, model_1]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                if i < 1:
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    # if i % 1 == 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
                    running_loss = 0.0
        net_full_save_path = os.path.join(args.get_default_model_folder_path(), str(model_id) + "_trained_Cifar10FC.model")
        torch.save(net.state_dict(),net_full_save_path)
        model_id += 1
        print('Finished Training for model '+ str(model_id))
def rand_color():
    dummy_image = []
    ar = []
    for h in range(32):
        for w in range(32):
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)

            ar.append((r,g,b))
        dummy_image.append(ar)
        ar = []
    ar = np.array(dummy_image)
    return ar

dummy_image = rand_color()
dummy_image = np.array(dummy_image.astype(np.uint8))
dummy_image = np.transpose(dummy_image, (2, 0, 1))
plt.imshow(np.transpose(dummy_image, (1, 2, 0)))
plt.show()
