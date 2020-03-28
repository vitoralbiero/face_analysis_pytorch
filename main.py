from config import Config
from train import Train
import argparse
import torch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a race, gender, or age predictor.')
    parser.add_argument('-e', '--epochs', help='Number of epochs.', default=30, type=int)
    parser.add_argument('-n', '--net', help='Residual type [ir, ir_se].', default='ir_se', type=str)
    parser.add_argument('-d', '--depth', help='Number of layers [50, 100, 152].', default=50, type=int)
    parser.add_argument('-lr', '--lr', help='Learning rate.', default=1e-3, type=float)
    parser.add_argument('-b', '--batch_size', help='Batch size.', default=512, type=int)
    parser.add_argument('-w', '--workers', help='Workers number.', default=4, type=int)
    parser.add_argument('-m', '--multi_gpu', help='Use multi gpus.', type=bool, default=True)

    parser.add_argument('-t', '--train_list', help='List of images to train.')
    parser.add_argument('-v', '--val_list', help='List of images to validate.')
    parser.add_argument('-ts', '--train_source', help='Path to the train images.')
    parser.add_argument('-vs', '--val_source', help='Path to the val images.')
    parser.add_argument('-a', '--attribute', help='Which attribute to train [race, gender, age].', type=str)
    parser.add_argument('-p', '--prefix', help='Prefix to save the model.', type=str)

    parser.add_argument('-pt', '--pretrained', help='Path to pretrained weights.', type=str)
    parser.add_argument('-r', '--resume', help='Path to load model to resume training.', type=str)

    args = parser.parse_args()

    config = Config(args.prefix, args.attribute.lower())

    config.net = args.net
    config.depth = args.depth
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.workers = args.workers
    config.train_list = args.train_list
    config.val_list = args.val_list
    config.train_source = args.train_source
    config.val_source = args.val_source
    config.epochs = args.epochs
    config.multi_gpu = args.multi_gpu
    config.pretrained = args.pretrained
    config.resume = args.resume

    torch.manual_seed(0)
    np.random.seed(0)

    train = Train(config)
    train.run()
