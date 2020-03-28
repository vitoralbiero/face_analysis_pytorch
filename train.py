from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from tqdm import tqdm
from model.resnet import ResNet
from utils.model_loader import save_state, load_state
from utils.utils import separate_bn_param
from torch import optim
from data.data_loader import CustomDataLoader
from model.gender_head import GenderHead
from model.age_head import AgeHead
import torch
from os import path
import numpy as np
from sklearn.metrics import mean_absolute_error


class Train():
    ATTR_HEAD = {'race': AgeHead, 'gender': GenderHead, 'age': AgeHead}

    def __init__(self, config):
        self.config = config
        print(self.config)
        self.save_file(self.config, 'config.txt')

        self.writer = SummaryWriter(config.log_path)

        self.model = ResNet(self.config.net_depth, self.config.drop_ratio, self.config.net_mode)
        self.head = self.ATTR_HEAD[self.config.attribute]()

        if self.config.pretrained:
            print(f'Loading pretrained weights from {self.config.pretrained}')
            load_state(self.model, self.head, None, self.config, self.config.pretrained, True)

        paras_only_bn, paras_wo_bn = separate_bn_param(self.model)

        dummy_input = torch.zeros(1, 3, 112, 112)
        self.writer.add_graph(self.model, dummy_input)

        if torch.cuda.device_count() > 1:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)
            self.head = DataParallel(self.head)

        self.model = self.model.to(self.config.device)
        self.head = self.head.to(self.config.device)

        self.train_loader = CustomDataLoader(self.config, self.config.train_source, self.config.train_list)

        if self.config.attribute != 'recognition':
            self.val_loader = CustomDataLoader(self.config, self.config.val_source,
                                               self.config.val_list, False, False, False)

        self.optimizer = optim.SGD([{'params': paras_wo_bn,
                                     'weight_decay': self.config.weight_decay},
                                    {'params': self.head.parameters(),
                                     'weight_decay': self.config.weight_decay},
                                    {'params': paras_only_bn}],
                                   lr=self.config.lr, momentum=self.config.momentum)

        if self.config.resume:
            print(f'Resuming training from {self.config.resume}')
            load_state(self.model, self.head, self.optimizer, self.config, self.config.resume, False)

        print(self.optimizer)
        self.save_file(self.optimizer, 'optimizer.txt')
        self.tensorboard_loss_every = 100
        self.evaluate_every = 2000
        self.save_every = 2000

        if self.config.attribute == 'recognition':
            self.agedb_30, self.agedb_30_issame = get_val_pair(self.config.val_source, 'agedb_30')
            self.cfp_fp, self.cfp_fp_issame = get_val_pair(self.config.val_source, 'cfp_fp')
            self.lfw, , self.lfw_issame = get_val_pair(self.config.val_source, 'lfw')

    def run(self):
        self.model.train()
        self.head.train()
        running_loss = 0.
        step = 0
        val_acc = 0.
        val_loss = 0.

        val_acc, val_loss = self.evaluate()

        for epoch in range(self.config.epochs):
            if epoch in self.config.reduce_lr:
                self.reduce_lr()

            loop = tqdm(iter(self.train_loader))
            for imgs, labels in loop:
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)

                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                outputs = self.head(embeddings)

                loss = self.config.loss(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                if step % self.tensorboard_loss_every == 0 and step != 0:
                    loss_board = running_loss / self.tensorboard_loss_every
                    self.writer.add_scalar('train_loss', loss_board, step)
                    running_loss = 0.

                if step % self.evaluate_every == 0 and step != 0:
                    val_acc, val_loss = self.evaluate()
                    self.model.train()
                    self.head.train()

                if step % self.save_every == 0 and step != 0:
                    save_state(self.model, self.head, self.optimizer, self.config, val_acc, False, step)

                step += 1
                loop.set_description('Epoch {}/{}'.format(epoch + 1, self.config.epochs))
                loop.set_postfix(loss=loss.item(), val_acc=val_acc, val_loss=val_loss)

        val_acc, val_loss = self.evaluate()
        self.tensorboard_val(val_acc, val_loss, step)
        save_state(self.model, self.head, self.optimizer, self.config, val_acc, True, step)

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def tensorboard_val(self, accuracy, loss, step):
        self.writer.add_scalar('val_acc', accuracy, step)

        if val_loss != 0:
            self.writer.add_scalar('val_loss', loss, step)

    def evaluate(self):
        if self.config.attribute != 'recognition':
            val_acc, val_loss = self.evaluate_attribute()
            self.tensorboard_val(val_acc, val_loss, step)

        elif self.config.attribute == 'recognition':
            return self.evaluate_recognition()

        return val_acc, val_loss

    def evaluate_attribute(self):
        self.model.eval()
        self.head.eval()

        y_true = torch.tensor([], dtype=torch.long, device=self.config.device)
        all_outputs = torch.tensor([], device=self.config.device)

        with torch.no_grad():
            for imgs, labels in iter(self.val_loader):
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device).long()

                embeddings = self.model(imgs)
                outputs = self.head(embeddings)

                y_true = torch.cat((y_true, labels), 0)
                all_outputs = torch.cat((all_outputs, outputs), 0)

            loss = self.config.loss(all_outputs, y_true).item()

        y_true = y_true.cpu().numpy()

        if self.config.attribute == 'age':
            y_pred = all_outputs.cpu().numpy()
            accuracy = mean_absolute_error(y_true, y_pred)
        else:
            _, y_pred = torch.max(all_outputs, 1)
            y_pred = y_pred.cpu().numpy()

            accuracy = round(np.sum(y_true == y_pred) / len(y_pred), 4)

        print(accuracy, loss)

        return round(accuracy, 4), round(loss, 4)

    def evaluate_recognition(self, samples, issame, nrof_folds=10, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(samples), self.config.embedding_size])

        with torch.no_grad():
            while idx + self.config.batch_size <= len(samples):
                batch = torch.tensor(samples[idx:idx + self.config.batch_size])
                embeddings[idx:idx + self.config.batch_size] = self.model(batch.to(self.config.device)).cpu()
                idx += self.config.batch_size

            if idx < len(samples):
                batch = torch.tensor(samples[idx:])
                embeddings[idx:] = self.model(batch.to(self.config.device)).cpu()

        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)

        return accuracy.mean(), 0

    def save_file(self, string, file_name):
        file = open(path.join(self.config.work_path, file_name), "w")
        file.write(str(string))
        file.close()
