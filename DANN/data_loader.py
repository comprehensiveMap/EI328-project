import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets
from scipy.io import loadmat
import scipy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import numpy as np

BATCH_SIZE = 128
scaler = MinMaxScaler()
num_repeat = 1
num_train = 10


class sentimentDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        self.len = data.shape[0]
        
    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor
    
    def __len__(self):
        return self.len


def load_data(batch_size=BATCH_SIZE, person=1):
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for i in range(1, num_train+1):
        mat_data = loadmat("../train/"+str(i)+".mat")
        train_data_list.append(scaler.fit_transform(mat_data['de_feature']))
        train_label_list.append(mat_data['label'])

    for i in range(11, 14):
        mat_data = loadmat("../test/"+str(i)+".mat")
        test_data_list.append(scaler.fit_transform(mat_data['de_feature']))
        test_label_list.append(mat_data['label'])
        for _ in range(num_repeat-1):
            test_data_list[i-11] = np.concatenate((test_data_list[i-11], scaler.fit_transform(mat_data['de_feature'])))
            test_label_list[i-11] = np.concatenate((test_label_list[i-11], mat_data['label']))

    train_datas = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_label_list)

    trainset = sentimentDataset(train_datas, train_labels)
    dataloader_source = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    testsets = [sentimentDataset(test_data_list[i], test_label_list[i]) for i in range(3)]
    dataloaders_target = [DataLoader(testset, batch_size=batch_size, shuffle=False) for testset in testsets]

    return dataloader_source, dataloaders_target[person-1]


def load_test_data(tar = True, batch_size=BATCH_SIZE, person=1):
    if tar:
        test_data_list = []
        test_label_list = []
        for i in range(11, 14):
            mat_data = loadmat("../test/"+str(i)+".mat")
            test_data_list.append(scaler.fit_transform(mat_data['de_feature']))
            test_label_list.append(mat_data['label'])
        testsets = [sentimentDataset(test_data_list[i], test_label_list[i]) for i in range(3)]
        dataloaders_target = [DataLoader(testset, batch_size=batch_size, shuffle=False) for testset in testsets]
        dataloader = dataloaders_target[person-1]
    else:
        train_data_list = []
        train_label_list = []
        for i in range(1, num_train+1):
            mat_data = loadmat("../train/"+str(i)+".mat")
            train_data_list.append(scaler.fit_transform(mat_data['de_feature']))
            train_label_list.append(mat_data['label'])
        train_datas = np.concatenate(train_data_list)
        train_labels = np.concatenate(train_label_list)
        trainset = sentimentDataset(train_datas, train_labels)
        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    return dataloader