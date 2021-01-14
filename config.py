from easydict import EasyDict
from os import path, makedirs
import torch
from torch.nn.functional import nll_loss, cross_entropy
from torch.nn import BCELoss
from metrics.metrics import AdaCos
from metrics.metrics import ArcFace
from metrics.metrics import CosFace
from metrics.metrics import SphereFace


class Config(EasyDict):
    LOSS = {'race': nll_loss, 'gender': nll_loss,
            'age': BCELoss(reduction='sum'), 'recognition': cross_entropy}
    MAX_OR_MIN = {'race': 'max', 'gender': 'max', 'age': 'min', 'recognition': 'max'}
    OUTPUT_TYPE = {'race': torch.long, 'gender': torch.long, 'age': torch.float, 'recognition': torch.long}
    RECOGNITION_HEAD = {'arcface': ArcFace, 'cosface': CosFace, 'adacos': AdaCos, 'sphereface': SphereFace}

    def __init__(self, prefix, attribute, recognition_head=None):
        self.prefix = prefix
        self.work_path = path.join('./workspace/', self.prefix)
        self.model_path = path.join(self.work_path, 'models')
        self.create_path(self.model_path)
        self.log_path = path.join(self.work_path, 'log')
        self.create_path(self.log_path)
        self.attribute = attribute
        self.loss = self.LOSS[attribute]
        self.input_size = [112, 112]
        self.embedding_size = 512
        self.depth = 50
        self.drop_ratio = 0.4
        self.net_mode = 'ir_se'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = False
        self.batch_size = 512
        self.weight_decay = 5e-4
        self.lr = 1e-3
        self.momentum = 0.9
        self.pin_memory = True
        self.frequency_log = 20
        self.epochs = 20
        self.reduce_lr = [9, 12, 14]
        self.lr_plateau = None
        self.early_stop = None
        self.workers = 4
        self.train_list = None
        self.train_source = None
        self.val_list = None
        self.val_source = None
        self.pretrained = None
        self.resume = None
        self.max_or_min = self.MAX_OR_MIN[attribute]
        self.output_type = self.OUTPUT_TYPE[attribute]
        self.recognition_head = None
        if recognition_head:
            self.recognition_head = self.RECOGNITION_HEAD[recognition_head.lower()]
        self.margin = None

    def create_path(self, file_path):
        if not path.exists(file_path):
            makedirs(file_path)
