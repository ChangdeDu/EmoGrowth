import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,TensorDataset
from torchvision import transforms
from tqdm import tqdm
import torch

class DataManager(object):
    def __init__(self, dataset_name,init_cls, increment,data_all):
        self.dataset_name = dataset_name
        self._increments = [init_cls]
        if self.dataset_name == 'iScience':
            while sum(self._increments) + increment <= 27:
                self._increments.append(increment)
        elif self.dataset_name == 'Neuroimage':
            while sum(self._increments) + increment <= 80:
                self._increments.append(increment)
        elif self.dataset_name == 'PNAS':
            while sum(self._increments) + increment <= 28:
                self._increments.append(increment)
        self.data_all = data_all
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[:task + 1])

    def get_total_classnum(self):
        if self.dataset_name == 'iScience':
            return 27
        elif self.dataset_name == 'Neuroimage':
            return 80

    def get_dataset(self, task_now, source,appendent=None,ret_data=False,affective=False):
        sub_data,label,affective_dimension = self.data_all
        label_now = [label[element[task_now]][:] for element in label['label_session']]
        train_label = np.transpose(label_now[0]).astype('float32')
        train_index = np.transpose(label_now[1])
        test_label = np.transpose(label_now[2]).astype('float32')
        test_index = np.transpose(label_now[3])
        if source == "train":
            train_x = torch.from_numpy(sub_data[np.int64(np.squeeze(train_index-1))])
            train_y = torch.from_numpy(train_label)
            train_affective_dimension = torch.from_numpy(affective_dimension[np.int64(np.squeeze(train_index-1))])
            if appendent is not None:
                appendent_data, appendent_targets_ori = appendent
                appendent_data = torch.from_numpy(appendent_data)
                appendent_targets = []
                for temp in appendent_targets_ori:
                    multi_hot_vector = np.zeros(self.get_accumulate_tasksize(task_now))
                    multi_hot_vector[temp] = 1
                    appendent_targets.append(list(multi_hot_vector))
                appendent_targets = torch.from_numpy(np.array(appendent_targets))
                train_x = torch.cat((train_x,appendent_data),dim=0)
                train_y = torch.hstack((torch.zeros([train_y.shape[0], self.get_accumulate_tasksize(task_now-1)]), train_y))
                train_y = torch.cat((train_y,appendent_targets),dim=0)
            print('train_samples_all = ', train_x.shape[0])
            if ret_data:
                if affective == True:
                    return train_x,train_y,train_affective_dimension,TensorDataset(train_x, train_y)
                else:
                    return train_x,train_y,TensorDataset(train_x, train_y)
            else:
                return TensorDataset(train_x, train_y)
        elif source == "test":
            print('test_sampes = ', test_index.shape[0])
            test_x = torch.from_numpy(sub_data[np.int64(np.squeeze(test_index-1))])
            test_y = torch.from_numpy(test_label)
            return TensorDataset(test_x, test_y)
        else:
            raise ValueError("Unknown data source {}.".format(source))