import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net_ml import IncrementalNet_AGCN
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.metrics import *
from utils.loss import LinkPredictionLoss_cosine
from torch.utils.data import Dataset,TensorDataset
### iscience ###
init_epoch = 20
init_lr = 0.0001
init_weight_decay = 0.01


epochs = 10
lrate = 0.0001
batch_size = 128
weight_decay = 0.01
num_workers = 8
lamda_kd_logits = 1

### pnas ###
# init_epoch = 40
# init_lr = 0.001
# init_weight_decay = 0
#
#
# epochs = 30
# lrate = 0.001
# batch_size = 128
# weight_decay = 0
# num_workers = 8
# lamda_kd_logits = 1



class AGCN(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet_AGCN(args)
        self.subject = args["subject"]
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        self._old_label_adj = self.label_adj

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_x,train_y,train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train",
            ret_data=True
        )
        ###calculate expert output and label_adj###
        if self._cur_task > 0:
            soft_label_known = self._old_network(train_x.to(self._device), self.label_adj.to(self._device))
            self.label_adj,self.soft_label = self.sym_conditional_prob_update(soft_label_known.cpu(), self.label_adj, train_y, ld=False)

            self.train_loader = DataLoader(
                TensorDataset(train_x,train_y), batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            ######
        else:
            self.label_adj = self.sym_conditional_prob(train_y)
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )

        test_dataset = data_manager.get_dataset(
            self._cur_task, source="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

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
                label_adj = self.label_adj.to(self._device)
                logits = self._network(inputs,label_adj)
                loss_clf = cost(logits, targets)
                loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map,train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,agcn=True)
            test_map,test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,agcn=True)

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
        trans = torch.nn.Sigmoid()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)


                label_adj = self.label_adj.to(self._device)
                logits = self._network(inputs,label_adj)

                fake_targets = targets
                loss_clf = cost(
                    logits[:, self._known_classes :], fake_targets
                )
                self._old_label_adj = self._old_label_adj.to(self._device)
                loss_kd_logits = cost(logits[:,:self._known_classes],trans(self._old_network(inputs,self._old_label_adj)))


                loss = loss_clf + lamda_kd_logits * loss_kd_logits
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,train=True,agcn=True)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,agcn=True)
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

    def fake_target_gen(self,targets):
        return torch.hstack((torch.zeros([targets.shape[0],self._known_classes]).to(self._device),targets))
