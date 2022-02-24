from os import makedirs, path

import torch
from easydict import EasyDict
from torch.nn import BCELoss, NLLLoss
from torch.nn.functional import cross_entropy

from metrics.metrics import AdaCos, ArcFace, CosFace, SphereFace


class Config(EasyDict):
    LOSS = {
        "race": NLLLoss,
        "gender": NLLLoss,
        "age": BCELoss(reduction="sum"),
        "recognition": cross_entropy,
    }
    MAX_OR_MIN = {"race": "max", "gender": "max", "age": "min", "recognition": "max"}
    OUTPUT_TYPE = {
        "race": torch.long,
        "gender": torch.long,
        "age": torch.float,
        "recognition": torch.long,
    }
    RECOGNITION_HEAD = {
        "arcface": ArcFace,
        "cosface": CosFace,
        "adacos": AdaCos,
        "sphereface": SphereFace,
    }

    def __init__(self, args):
        self.prefix = args.prefix
        self.work_path = path.join("./workspace/", self.prefix)
        self.model_path = path.join(self.work_path, "models")
        self.create_path(self.model_path)
        self.log_path = path.join(self.work_path, "log")
        self.create_path(self.log_path)
        self.attribute = args.attribute.lower()
        self.loss = self.LOSS[self.attribute]
        self.input_size = [112, 112]
        self.embedding_size = 512
        self.depth = args.depth
        self.drop_ratio = 0.4
        self.net_mode = args.net_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = args.multi_gpu
        self.batch_size = args.batch_size
        self.weight_decay = 5e-4
        self.lr = args.lr
        self.momentum = 0.9
        self.pin_memory = True
        self.frequency_log = 20
        self.epochs = args.epochs
        self.reduce_lr = [9, 12, 14]
        self.lr_plateau = args.lr_plateau
        self.early_stop = args.early_stop
        self.workers = args.workers
        self.train_list = args.train_list
        self.train_source = args.train_source
        self.val_list = args.val_list
        self.val_source = args.val_source
        self.pretrained = args.pretrained
        self.resume = args.resume
        self.max_or_min = self.MAX_OR_MIN[self.attribute]
        self.output_type = self.OUTPUT_TYPE[self.attribute]
        self.recognition_head = None
        if args.head:
            self.recognition_head = self.RECOGNITION_HEAD[args.head.lower()]
        self.margin = args.margin
        self.use_mask = args.use_mask

    def create_path(self, file_path):
        if not path.exists(file_path):
            makedirs(file_path)
