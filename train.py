import argparse
import copy
import itertools
from os import path

import higher
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import autograd, optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.data_loader_train_lmdb import LMDBDataLoader
from data.load_test_sets_recognition import get_val_pair
from model.age_head import AgeHead
from model.gender_head import GenderHead
from model.model_wrapper import ModelWrapper
from model.race_head import RaceHead
from model.resnet import ResNet
from optimizer.early_stop import EarlyStop
from recognition import verification
from utils.model_loader import load_state, save_state
from utils.train_logger import TrainLogger
from utils.utils import separate_bn_param


class Train:
    def __init__(self, config):
        self.config = config

        ATTR_HEAD = {
            "race": RaceHead,
            "gender": GenderHead,
            "age": AgeHead,
            "recognition": self.config.recognition_head,
        }

        self.writer = SummaryWriter(config.log_path)

        if path.isfile(self.config.train_source):
            self.train_loader = LMDBDataLoader(
                config=self.config,
                lmdb_path=self.config.train_source,
                train=True,
                use_mask=self.config.use_mask,
            )

        class_num = self.train_loader.class_num()
        print(len(self.train_loader.dataset))
        print(f"Classes: {class_num}")

        self.model = ResNet(
            self.config.depth, self.config.drop_ratio, self.config.net_mode
        )
        if self.config.attribute == "recognition":
            self.head = ATTR_HEAD[self.config.attribute](
                classnum=class_num, m=self.config.margin
            )
        else:
            self.head = ATTR_HEAD[self.config.attribute](classnum=class_num)

        self.full_model = ModelWrapper(self.model, self.head).to(self.config.device)

        paras_only_bn, paras_wo_bn = separate_bn_param(self.model)

        dummy_input = torch.zeros(1, 3, 112, 112).to(self.config.device)
        self.writer.add_graph(self.full_model, dummy_input)

        if torch.cuda.device_count() > 1:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.full_model = DataParallel(self.full_model)

        self.weights = None
        if self.config.attribute in ["race", "gender"]:
            _, self.weights = np.unique(
                self.train_loader.dataset.get_targets(), return_counts=True
            )
            self.weights = np.max(self.weights) / self.weights
            self.weights = torch.tensor(
                self.weights, dtype=torch.float, device=self.config.device
            )
            self.config.weights = self.weights
            print(self.weights)

            self.config.loss = self.config.loss(weight=self.weights)

        if self.config.val_source is not None:
            if self.config.attribute != "recognition":
                if path.isfile(self.config.val_source):
                    self.val_loader = LMDBDataLoader(
                        config=self.config,
                        lmdb_path=self.config.val_source,
                        train=False,
                        use_mask=self.config.use_mask,
                    )

            else:
                self.validation_list = []
                for val_name in config.val_list:
                    dataset, issame = get_val_pair(self.config.val_source, val_name)
                    self.validation_list.append([dataset, issame, val_name])

        self.optimizer = optim.SGD(
            [
                {"params": paras_wo_bn, "weight_decay": self.config.weight_decay},
                {
                    "params": self.head.parameters(),
                    "weight_decay": self.config.weight_decay,
                },
                {"params": paras_only_bn},
            ],
            lr=self.config.lr,
            momentum=self.config.momentum,
        )

        if self.config.resume:
            print(f"Resuming training from {self.config.resume}")
            load_state(self.full_model, self.optimizer, self.config.resume, False)

        if self.config.pretrained:
            print(f"Loading pretrained weights from {self.config.pretrained}")
            load_state(
                full_model=self.full_model,
                optimizer=None,
                path_to_model=self.config.pretrained,
                model_only=True,
                load_head=self.config.attribute != "recognition",
            )

        print(self.config)
        self.save_file(self.config, "config.txt")

        print(self.optimizer)
        self.save_file(self.optimizer, "optimizer.txt")

        self.tensorboard_loss_every = max(len(self.train_loader) // 100, 1)
        self.evaluate_every = max(len(self.train_loader) // 5, 1)

        if self.config.lr_plateau:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.max_or_min,
                factor=0.1,
                patience=3,
                verbose=True,
                threshold=0.001,
                cooldown=1,
            )
        if self.config.early_stop:
            self.early_stop = EarlyStop(mode=self.config.max_or_min)

    def run(self):
        self.full_model.train()
        running_loss = 0.0
        step = 0
        val_acc = 0.0

        best_step = 0
        best_acc = float("Inf")
        if self.config.max_or_min == "max":
            best_acc *= -1

        for epoch in range(self.config.epochs):
            train_logger = TrainLogger(
                self.config.batch_size, self.config.frequency_log
            )

            if epoch + 1 in self.config.reduce_lr and not self.config.lr_plateau:
                self.reduce_lr()

            for idx, data in enumerate(self.train_loader):
                imgs, labels = data
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)

                self.optimizer.zero_grad()

                if self.config.attribute == "recognition":
                    outputs = self.full_model(imgs, labels)
                else:
                    outputs = self.full_model(imgs)

                loss = self.config.loss(outputs, labels)

                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                if step % self.tensorboard_loss_every == 0:
                    loss_board = running_loss / self.tensorboard_loss_every
                    self.writer.add_scalar("train_loss", loss_board, step)
                    running_loss = 0.0

                if step % self.evaluate_every == 0 and step != 0:
                    if self.config.val_source is not None:
                        val_acc, _ = self.evaluate(step)
                        self.full_model.train()
                        best_acc, best_step = self.save_model(
                            val_acc, best_acc, step, best_step
                        )
                        print(f"Best accuracy: {best_acc:.5f} at step {best_step}")
                    else:
                        save_state(
                            self.full_model, self.optimizer, self.config, 0, step
                        )

                train_logger(
                    epoch, self.config.epochs, idx, len(self.train_loader), loss.item()
                )
                step += 1

            if self.config.lr_plateau:
                self.scheduler.step(val_acc)

            if self.config.early_stop:
                self.early_stop(val_acc)
                if self.early_stop.stop:
                    print("Early stopping model...")
                    break

        val_acc, val_loss = self.evaluate(step)
        best_acc = self.save_model(val_acc, best_acc, step, best_step)
        print(f"Best accuracy: {best_acc} at step {best_step}")

    def save_model(self, val_acc, best_acc, step, best_step):
        if (self.config.max_or_min == "max" and val_acc > best_acc) or (
            self.config.max_or_min == "min" and val_acc < best_acc
        ):
            best_acc = val_acc
            best_step = step
            save_state(self.full_model, self.optimizer, self.config, val_acc, step)

        return best_acc, best_step

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

        print(self.optimizer)

    def tensorboard_val(self, accuracy, step, loss=0, dataset=""):
        self.writer.add_scalar("{}val_acc".format(dataset), accuracy, step)

        if self.config.attribute != "recognition":
            self.writer.add_scalar("val_loss", loss, step)

    def evaluate(self, step):
        if self.config.attribute != "recognition":
            val_acc, val_loss = self.evaluate_attribute()
            self.tensorboard_val(val_acc, step, val_loss)

        elif self.config.attribute == "recognition":
            val_loss = 0
            val_acc = 0
            print("Validating...")
            for idx, validation in enumerate(self.validation_list):
                dataset, issame, val_name = validation
                acc, std = self.evaluate_recognition(dataset, issame)
                self.tensorboard_val(acc, step, dataset=f"{val_name}_")
                print(f"{val_name}: {acc:.5f}+-{std:.5f}")
                val_acc += acc

            val_acc /= idx + 1
            self.tensorboard_val(val_acc, step)
            print(f"Mean accuracy: {val_acc:.5f}")

        return val_acc, val_loss

    def evaluate_attribute(self):
        self.full_model.eval()

        y_true = torch.tensor(
            [], dtype=self.config.output_type, device=self.config.device
        )
        all_outputs = torch.tensor([], device=self.config.device)

        with torch.no_grad():
            for imgs, labels in iter(self.val_loader):
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.full_model(imgs)

                y_true = torch.cat((y_true, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

            if self.weights is not None:
                loss = round(self.config.loss(all_outputs, y_true).item(), 4)
            else:
                loss = round(self.config.loss(all_outputs, y_true).item(), 4)

        y_true = y_true.cpu().numpy()

        if self.config.attribute == "age":
            y_pred = all_outputs.cpu().numpy()
            y_pred = np.round(y_pred, 0)
            y_pred = np.sum(y_pred, axis=1)
            y_true = np.sum(y_true, axis=1)
            accuracy = round(mean_absolute_error(y_true, y_pred), 4)
        else:
            _, y_pred = torch.max(all_outputs, 1)
            y_pred = y_pred.cpu().numpy()

            accuracy = round(np.sum(y_true == y_pred) / len(y_pred), 4)

        return accuracy, loss

    def evaluate_recognition(self, samples, issame, nrof_folds=10, tta=False):
        self.full_model.eval()
        idx = 0
        embeddings = np.zeros([len(samples), self.config.embedding_size])

        with torch.no_grad():
            for idx in range(0, len(samples), self.config.batch_size):
                batch = torch.tensor(samples[idx : idx + self.config.batch_size])
                embeddings[
                    idx : idx + self.config.batch_size
                ] = self.full_model.module.model(batch.to(self.config.device)).cpu()
                idx += self.config.batch_size

        tpr, fpr, accuracy, best_thresholds = verification.evaluate(
            embeddings, issame, nrof_folds
        )

        return round(accuracy.mean(), 5), round(accuracy.std(), 5)

    def save_file(self, string, file_name):
        file = open(path.join(self.config.work_path, file_name), "w")
        file.write(str(string))
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a race, gender, age or recognition models."
    )

    # network and training parameters
    parser.add_argument(
        "--epochs", "-e", help="Number of epochs.", default=30, type=int
    )
    parser.add_argument(
        "--net_mode", "-n", help="Residual type [ir, ir_se].", default="ir_se", type=str
    )
    parser.add_argument(
        "--depth", "-d", help="Number of layers [50, 100, 152].", default=50, type=int
    )
    parser.add_argument("--lr", "-lr", help="Learning rate.", default=0.001, type=float)
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=384, type=int)
    parser.add_argument(
        "--lr_plateau", "-lrp", help="Reduce lr on plateau.", action="store_true"
    )
    parser.add_argument(
        "--early_stop", "-es", help="Use early stop.", action="store_true"
    )
    parser.add_argument(
        "--multi_gpu", "-m", help="Use multi gpus.", action="store_true"
    )
    parser.add_argument("--workers", "-w", help="Workers number.", default=4, type=int)
    parser.add_argument(
        "--num_classes", "-nc", help="Number of classes.", default=85742, type=int
    )

    # training/validation configuration
    parser.add_argument("--train_list", "-t", help="List of images to train.")
    parser.add_argument(
        "--val_list",
        "-v",
        help="List of images to validate, or datasets to validate (recognition).",
        default=["agedb_30", "cfp_fp", "lfw"],
    )
    parser.add_argument(
        "--train_source", "-ts", help="Path to the train images, or dataset LMDB file."
    )
    parser.add_argument(
        "--val_source", "-vs", help="Path to the val images, or dataset LMDB file."
    )
    parser.add_argument(
        "--attribute",
        "-a",
        help="Which attribute to train [race, gender, age, recognition].",
        type=str,
    )
    parser.add_argument(
        "--head",
        "-hd",
        help="If recognition, which head to use [arcface, cosface, adacos].",
        type=str,
    )
    parser.add_argument("--margin", "-margin", help="Margin", default=0.5, type=float)
    parser.add_argument("--prefix", "-p", help="Prefix to save the model.", type=str)

    # resume from or load pretrained weights
    parser.add_argument(
        "--pretrained", "-pt", help="Path to pretrained weights.", type=str
    )
    parser.add_argument(
        "--resume", "-r", help="Path to load model to resume training.", type=str
    )

    # use masks to focus on face
    parser.add_argument(
        "--use_mask", "-us", help="Mask images with masks.", action="store_true"
    )

    args = parser.parse_args()

    config = Config(args)

    torch.manual_seed(0)
    np.random.seed(0)

    train = Train(config)
    train.run()
