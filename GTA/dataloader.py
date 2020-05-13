import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data

def datasetLoad(batch_size = 64, person = 1):
    X_source = np.array([])
    y_source = np.array([])

    for i in range(10):
        data = scipy.io.loadmat('../train/%d.mat'%(i+1))['de_feature']
        label = scipy.io.loadmat('../train/%d.mat'%(i+1))['label']
        
        if i == 0:
            X_source = data
            y_source = label
        else:
            X_source = np.vstack((X_source, data))
            y_source = np.vstack((y_source, label))


    X_source =  -1 + 2 * (X_source - np.min(X_source, axis=0)) / (np.max(X_source, axis=0) - np.min(X_source, axis=0))
    X_source, X_val, y_source, y_val = train_test_split(X_source, y_source, test_size = 0.1, random_state = 0)
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).long().squeeze()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long().squeeze()
    

    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)
    sourceval_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    X_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_target = -1 + 2 * (X_target - np.min(X_target, axis=0)) / (np.max(X_target, axis=0) - np.min(X_target, axis=0))
    X_target = torch.from_numpy(X_target).float()
    y_target = torch.from_numpy(y_target).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)

    return source_dataset, sourceval_dataset, target_dataset