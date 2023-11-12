
import torch.nn.functional as F
import torch
import json
import time

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger
        self.debugOrRun = 'debug_results' # 'debug_results', 'results'
        self.dataset = 'mnist' # 'cifar100', 'lfw', 'mnist', 'celebA'
        self.set_imidx = 58486 # int or 000000
        self.net = 'lenet' # 'lenet', 'fc2', 'resnet'
        self.net_mt_diff = True
        self.batch_size = 1
        self.model_path = './model'
        self.root_path = '.'
        self.lr = 0.1
        self.use_game = False
        self.earlystop = 1e-9
        self.num_dummy = 1 # the number of reconstructed images
        self.iteration = 300
        self.num_exp = 1
        # self.methods = ['DLG', 'iDLG', 'mDLG', 'mDLG_mt', 'DLGAdam', 'InvG', 'CPA']
        self.methods = ['mDLG_mt']
        self.int_time = int(time.time())
        self.log_interval = 10


        self.train_data_loader_pickle_path = "data_loaders/cifar100/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/cifar100/test_data_loader.pickle"


    def get_logger(self):
        return self.logger

    def get_dataset(self):
        return self.dataset

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def get_num_dummy(self):
        return self.num_dummy

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path
    def get_default_model_folder_path(self):
        return self.model_path

    def get_root_path(self):
        return self.root_path

    def get_debugOrRun(self):
        return self.debugOrRun

    def get_lr(self):
        return self.lr

    def get_earlystop(self):
        return self.earlystop

    def get_iteration(self):
        return self.iteration

    def get_num_exp(self):
        return self.num_exp

    def get_methods(self):
        return self.methods

    def get_start_index_str(self):
        return self.start_index_str

    def get_log_interval(self):
        return self.log_interval

    def get_net(self):
        return self.net

    def get_net_mt_diff(self):
        return self.net_mt_diff

    def get_imidx(self):
        return self.set_imidx

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Iteration: {}\n".format(self.iteration) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Model Path (Relative): {}\n".format(self.model_path) + \
               "Methods: {}\n".format(self.methods) + \
               "Number Exp: {}\n".format(self.num_exp) + \
               "Dataset: {}\n".format(self.dataset) + \
               "Log Interval: {}\n".format(self.log_interval)

