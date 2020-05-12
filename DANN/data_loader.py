import torch.utils.data as data
import torch
from PIL import Image
import os
import scipy.io
import numpy as np

BATCH_SIZE = 128


def load_data(batch_size=128):
    X_train = np.array([])
    y_train = np.array([])

    for i in range(10):
        data = scipy.io.loadmat('../train/%d.mat'%(i+1))['de_feature']
        label = scipy.io.loadmat('../train/%d.mat'%(i+1))['label']
        
        if i == 0:
            X_train = data
            y_train = label
        else:
            X_train = np.vstack((X_train, data))
            y_train = np.vstack((y_train, label))

    X_train = (X_train - np.min(X_train, axis = 0)) / (np.max(X_train, axis = 0) - np.min(X_train, axis=0))
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long().squeeze()


    dataloader_source = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)


    X_test = scipy.io.loadmat('../test/%d.mat'%(11))['de_feature']
    y_test = scipy.io.loadmat('../test/%d.mat'%(11))['label']
    X_test = (X_test - np.min(X_test,axis=0)) / (np.max(X_test,axis=0) - np.min(X_test,axis=0))
    X_test = torch.from_numpy(X_test).float().repeat(10,1)
    y_test = torch.from_numpy(y_test).long().repeat(10,1).squeeze()

    dataloader_target = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)
    return dataloader_source, dataloader_target


def load_test_data(tar = True, batch_size=128):
    X = np.array([])
    y = np.array([])
    if tar:
        X = scipy.io.loadmat('../test/%d.mat'%(11))['de_feature']
        y = scipy.io.loadmat('../test/%d.mat'%(11))['label']   
    else:
        for i in range(10):
            data = scipy.io.loadmat('../train/%d.mat'%(i+1))['de_feature']
            label = scipy.io.loadmat('../train/%d.mat'%(i+1))['label']
            
            if i == 0:
                X = data
                y = label
            else:
                X = np.vstack((X, data))
                y = np.vstack((y, label))

    X = (X - np.min(X, axis = 0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long().squeeze()

    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=8)
    
    return dataloader