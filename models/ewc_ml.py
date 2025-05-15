import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.podnet import pod_spatial_loss
from utils.inc_net_ml import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.metrics import *


EPSILON = 1e-8
### iScience ###
# init_epoch = 30
# init_lr = 0.0001
# init_weight_decay = 0.01
#
# epochs = 10
# lrate = 0.0001
# batch_size = 128
# weight_decay = 0.001
# num_workers = 8
### PNAS ###
init_epoch = 45
init_lr = 0.001
# init_weight_decay = 0.01
init_weight_decay = 0

epochs = 40
lrate = 0.001
batch_size = 128
# weight_decay = 0.01
weight_decay = 0
num_workers = 8




T = 2
lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.fisher = None
        self._network = IncrementalNet(args)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train"
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            self._cur_task,source='test'
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                        alpha * self.fisher[n]
                        + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            self._init_train(train_loader, test_loader, optimizer)
        else:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=lrate,
                weight_decay=weight_decay,
            )
            self._update_representation(train_loader, test_loader, optimizer)

    def _init_train(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(init_epoch))
        cost = torch.nn.MultiLabelSoftMarginLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = cost(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map,train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader)
            test_map,test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Train_other_metrics {}, Test_other_metrics {}".format(
                self._cur_task,
                epoch + 1,
                init_epoch,
                losses / len(train_loader),
                train_map,
                test_map,
                train_other_metrics,
                test_other_metrics
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(epochs))
        cost = torch.nn.MultiLabelSoftMarginLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # fake_targets = targets
                # loss_clf = cost(
                #     logits[:, self._known_classes:], fake_targets
                # )

                fake_targets = self.fake_target_gen(targets)
                loss_clf = cost(
                    logits, fake_targets
                )

                loss_ewc = self.compute_ewc()
                loss = loss_clf + lamda * loss_ewc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,train=True)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Train_other_metrics {}, Test_other_metrics {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_map,
                test_map,
                train_other_metrics,
                test_other_metrics
            )
            prog_bar.set_description(info)
        logging.info(info)

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=lrate)
        cost = torch.nn.MultiLabelSoftMarginLoss()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = cost(logits[:, self._known_classes:], targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher

    def fake_target_gen(self,targets):
        return torch.hstack((torch.zeros([targets.shape[0],self._known_classes]).to(self._device),targets))