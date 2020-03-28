from easydict import EasyDict
from os import path, makedirs
import torch
from torch.nn.functional import nll_loss
from torch.nn import MSELoss


LOSS = {'race': nll_loss, 'gender': nll_loss, 'age': MSELoss()}


class Config(EasyDict):
    def __init__(self, prefix, attribute):
        self.prefix = prefix
        self.work_path = path.join('./work_space', self.prefix)
        self.model_path = path.join(self.work_path, 'models')
        self.create_path(self.model_path)
        self.log_path = path.join(self.work_path, 'log')
        self.create_path(self.log_path)
        self.save_path = path.join(self.work_path, 'final')
        self.create_path(self.save_path)
        self.attribute = attribute
        self.loss = LOSS[attribute]
        self.input_size = [112, 112]
        self.embedding_size = 512
        self.use_mobilfacenet = False
        self.net_depth = 50
        self.drop_ratio = 0.4
        self.net_mode = 'ir_se'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = False
        self.batch_size = 512
        self.weight_decay = 5e-4
        self.lr = 1e-3
        self.momentum = 0.9
        self.pin_memory = True
        self.epochs = 20
        self.reduce_lr = [12, 15, 18]
        self.workers = 4
        self.train_list = None
        self.train_source = None
        self.val_list = None
        self.val_source = None
        self.pretrained = None
        self.resume = None

    def create_path(self, file_path):
        if not path.exists(file_path):
            makedirs(file_path)
