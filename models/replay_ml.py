import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net_ml import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.metrics import *
### iScience ###
init_epoch = 30
init_lr = 0.0001
init_weight_decay = 0.01


epochs = 40
lrate = 0.0001
batch_size = 128
weight_decay = 0.001
num_workers = 8

### PNAS ###
# init_epoch = 45
# init_lr = 0.001
# # init_weight_decay = 0.01
# init_weight_decay = 0
#
# epochs = 40
# lrate = 0.001
# batch_size = 128
# # weight_decay = 0.01
# weight_decay = 0
# num_workers = 8

class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args)
        self.buffer_type = args['buffer_type']
        self.all_classes = args['total_class']

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Loader
        train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train",
            appendent=self._get_memory_ml(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            self._cur_task, source="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if self._total_classes < self.all_classes:
            if self.buffer_type == 'random':
                self.build_rehearsal_memory_ml(data_manager, self.samples_per_class)
            elif self.buffer_type == 'rs':
                self.build_rehearsal_memory_ml_rs(data_manager, self._memory_size)
            elif self.buffer_type == 'ocdm':
                self.build_rehearsal_memory_ml_ocdm(data_manager, self._memory_size)
            elif self.buffer_type == 'prs':
                self.build_rehearsal_memory_ml_prs(data_manager, self._memory_size)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
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
            )  # 1e-5
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


                fake_targets = targets
                loss_clf = cost(
                    logits, fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,train=True,er=True)
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
