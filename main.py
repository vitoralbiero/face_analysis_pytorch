from config import Config
from train import Train
import argparse
import torch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a race, gender, or age predictor.')
    parser.add_argument('--epochs', '-e', help='Number of epochs.', default=30, type=int)
    parser.add_argument('--net_mode', '-n', help='Residual type [ir, ir_se].', default='ir_se', type=str)
    parser.add_argument('--depth', '-d', help='Number of layers [50, 100, 152].', default=50, type=int)
    parser.add_argument('--lr', '-lr', help='Learning rate.', default=0.001, type=float)
    parser.add_argument('--batch_size', '-b', help='Batch size.', default=384, type=int)
    parser.add_argument('--lr_plateau', '-lrp', help='Reduce lr on plateau.', type=bool, default=True)
    parser.add_argument('--workers', '-w', help='Workers number.', default=4, type=int)
    parser.add_argument('--multi_gpu', '-m', help='Use multi gpus.', type=bool, default=True)

    parser.add_argument('--train_list', '-t', help='List of images to train.')
    parser.add_argument('--val_list', '-v', help='List of images to validate.')
    parser.add_argument('--train_source', '-ts', help='Path to the train images.')
    parser.add_argument('--val_source', '-vs', help='Path to the val images.')
    parser.add_argument('--attribute', '-a',
                        help='Which attribute to train [race, gender, age, recognition].', type=str)
    parser.add_argument('--head', '-hd',
                        help='If recognition, which head to use [arcface, cosface].', type=str)
    parser.add_argument('--prefix', '-p', help='Prefix to save the model.', type=str)

    parser.add_argument('--pretrained', '-pt', help='Path to pretrained weights.', type=str)
    parser.add_argument('--resume', '-r', help='Path to load model to resume training.', type=str)

    args = parser.parse_args()

    config = Config(args.prefix, args.attribute.lower(), args.head.lower())

    config.net_mode = args.net_mode
    config.depth = args.depth
    config.lr = args.lr
    config.lr_plateau = args.lr_plateau
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
