import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net_ml import IncrementalNet_CLIF
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.metrics import *
from utils.loss import LinkPredictionLoss_cosine
from torch.utils.data import Dataset,TensorDataset
### iscience ###

# # init_epoch = 20
# init_epoch = 50
# init_lr = 0.0001
# init_weight_decay = 0.005
#
#
# # epochs = 10
# lrate = 0.0001
# batch_size = 128
# weight_decay = 0.005
# num_workers = 8
# lamda_kd_logits = 1


## iscience ###
init_epoch = 40
init_lr = 0.0001
init_weight_decay = 0.005


epochs = 30
lrate = 0.0001
batch_size = 128
weight_decay = 0.005
num_workers = 8
lamda_kd_logits = 1

# ### pnas ###
# init_epoch = 50
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
# lamda_kd_logits = 1


# lamda_le = 0.005
# lamda_kd_relation = 1

class CLIF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet_CLIF(args)
        self.ld = args["ld"]
        self.lamda_kd_relation_aff = args['lamda_kd_relation_aff']
        self.lamda_kd_relation_data = args['lamda_kd_relation_data']
        self.lamda_le = args['lamda_le']
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

        train_x,train_y,train_affective_dimension,train_dataset = data_manager.get_dataset(
            self._cur_task,
            source="train",
            ret_data=True,
            affective=True
        )
        self.train_x_ld = train_x

        ###calculate expert output and label_adj###
        if self._cur_task > 0:
            soft_label_known,_ = self._old_network(train_x.to(self._device), self.label_adj.to(self._device))
            self.label_adj,self.soft_label = self.sym_conditional_prob_update(soft_label_known.cpu(), self.label_adj, train_y, ld=self.ld)
            # self.train_loader = DataLoader(
            #     TensorDataset(train_x,train_y,self.soft_label), batch_size=batch_size, shuffle=True, num_workers=num_workers
            # )
            ### 引入消歧计算标签共生但不用于计算知识蒸馏 args_ld is true###
            self.train_loader = DataLoader(
                TensorDataset(train_x,train_y,soft_label_known.cpu(),train_affective_dimension), batch_size=batch_size, shuffle=True, num_workers=num_workers
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
        emb_cost = LinkPredictionLoss_cosine()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                label_adj = self.label_adj.to(self._device)
                loss_adj = label_adj + torch.eye(label_adj.data.size(0), dtype=label_adj.data.dtype,device=label_adj.data.device) ###identity matrix included###
                logits,label_embedding = self._network(inputs,label_adj)
                loss_clf = cost(logits, targets)
                loss_le = emb_cost(label_embedding,loss_adj)
                loss = loss_clf + self.lamda_le * loss_le
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map,train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,clif=True)
            test_map,test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,clif=True)

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
        emb_cost = LinkPredictionLoss_cosine()
        trans = torch.nn.Sigmoid()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (inputs, targets,soft_targets,affective_dimension) in enumerate(train_loader):
                inputs, targets,soft_targets,affective_dimension = inputs.to(self._device), targets.to(self._device), soft_targets.to(self._device),affective_dimension.to(self._device)
                ### 引入消歧计算标签共生但不用于计算知识蒸馏 args_ld is true###
                soft_targets = trans(soft_targets)
                ######

                label_adj = self.label_adj.to(self._device)
                loss_adj = label_adj + torch.eye(label_adj.data.size(0), dtype=label_adj.data.dtype,device=label_adj.data.device) ###identity matrix included###
                logits,label_embedding = self._network(inputs,label_adj)

                ### 基于样本相似度的知识蒸馏 ###
                _, _, feature_old = self._old_network(inputs,self._old_label_adj.to(self._device),kd=True)
                logits_1, label_embedding_1, feature_new = self._network(inputs,label_adj,kd=True)
                ######

                fake_targets = targets
                loss_clf = cost(
                    logits[:, self._known_classes :], fake_targets
                )

                loss_le = emb_cost(label_embedding,loss_adj)
                loss_kd_logits = cost(logits[:,:self._known_classes],soft_targets)
                loss_kd_relation_1 = self.compute_relation_based_kd_loss(feature_old,feature_new)
                loss_kd_relation_2 = self.compute_relation_based_kd_loss(affective_dimension,feature_new)
                # loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_data * loss_kd_relation_1 + lamda_kd_logits * loss_kd_logits ### 删除情感维度蒸馏 ###
                # loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_aff * loss_kd_relation_2 + lamda_kd_logits * loss_kd_logits ### 删除样本蒸馏 ###
                # loss = loss_clf + self.lamda_le * loss_le + lamda_kd_logits * loss_kd_logits ### 删除relation蒸馏 ###
                loss = loss_clf + self.lamda_le * loss_le + self.lamda_kd_relation_data * loss_kd_relation_1 + self.lamda_kd_relation_aff * loss_kd_relation_2 + lamda_kd_logits * loss_kd_logits
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            train_map, train_other_metrics = self._compute_multi_label_accuracy(self._network, self.train_loader,clif=True,update=True,train=True)
            test_map, test_other_metrics = self._compute_multi_label_accuracy(self._network, self.test_loader,clif=True)
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

    def compute_relation_based_kd_loss(self,feature_old,feature_new):
        remove = 0
        feature_old = feature_old-torch.mean(feature_old,dim=0)
        feature_new = feature_new-torch.mean(feature_new,dim=0)
        RSM_old = torch.nn.functional.cosine_similarity(feature_old.unsqueeze(1), feature_old.unsqueeze(0), dim=-1)
        RSM_new = torch.nn.functional.cosine_similarity(feature_new.unsqueeze(1), feature_new.unsqueeze(0), dim=-1)
        if self.subject == 'visual':
            loss = (RSM_old - torch.diag_embed(torch.diag(RSM_old))) - (RSM_new - torch.diag_embed(torch.diag(RSM_new)))
        else:
            loss = torch.atanh(RSM_old-torch.diag_embed(torch.diag(RSM_old)))-torch.atanh(RSM_new-torch.diag_embed(torch.diag(RSM_new)))
        if len(torch.where(torch.isinf(loss))[0])>0:
            # print('inf appears.')
            remove = len(torch.where(torch.isinf(loss))[0])
            # print('remove=',remove)
            # print('position',torch.where(torch.isinf(loss)))
            cnn_f = RSM_old-torch.diag_embed(torch.diag(RSM_old))
            rsa = RSM_new-torch.diag_embed(torch.diag(RSM_new))
            # print(cnn_f[torch.where(torch.isinf(loss))])
            # print(rsa[torch.where(torch.isinf(loss))])
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, 0), loss)

        if len(torch.where(torch.isnan(loss))[0])>0:
            # print('nan appears.')
            remove = len(torch.where(torch.isnan(loss))[0])
            # print('remove=',remove)
            # print('position',torch.where(torch.isnan(loss)))
            cnn_f = RSM_old-torch.diag_embed(torch.diag(RSM_old))
            rsa = RSM_new-torch.diag_embed(torch.diag(RSM_new))
            # print(cnn_f[torch.where(torch.isnan(loss))])
            # print(rsa[torch.where(torch.isnan(loss))])
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
        loss = torch.pow(loss, 2)
        num = (feature_old.size(0)*(feature_old.size(0)-1))/2
        return loss.sum()/2/(num-remove//2)
