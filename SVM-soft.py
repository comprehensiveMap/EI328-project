# This file is an soft-voting mechanism one-vs-rest implementation of SVM, which reaches better accurcy than the default SVM.
# It will be trained for several minutes, kind of long.

import scipy.io
import numpy as np
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

def get_svm_one_to_one(data, label, i):
    m = SVC(C = 10, probability = True, cache_size = 4000, class_weight = {4:1, i:3}, gamma = 'auto')
    m.fit(data, label)
    
    return m

def get_model_one_to_other(y_train, X_train):
    model_list = []
    
    for i in [0,1,2,3]:
        y_train0 = y_train.flatten()
        y_train0[(y_train0 != i)] = 4
        model_list.append(get_svm_one_to_one(X_train, y_train0, i))
    
    return model_list

def one_to_other(model_list, y_test, X_test):
    predict = np.zeros([y_test.shape[0], 4])

    for i in range(4):
        pre_prob = model_list[i].predict_proba(X_test)
        labels = model_list[i].classes_
        if labels[0] != 4:
            predict[:,labels[0]] = np.array(pre_prob)[:,0]
        else:
            predict[:,labels[1]] = np.array(pre_prob)[:,1]
    
    predict_label = np.argmax(predict, axis=1)
    
    acc = sum(predict_label == y_test.flatten()) / y_test.shape[0]
    print("Ovr accuracy for one person: ", acc)
    
    return acc

X_train = np.array([])
y_train = np.array([])

for i in range(10):
    data = scipy.io.loadmat('train/%d.mat'%(i+1))['de_feature']
    label = scipy.io.loadmat('train/%d.mat'%(i+1))['label']
    
    if i == 0:
        X_train = data
        y_train = label
    else:
        X_train = np.vstack((X_train, data))
        y_train = np.vstack((y_train, label))

X_train = (X_train - np.min(X_train, axis = 0)) / (np.max(X_train, axis = 0) - np.min(X_train, axis=0))

model_list = get_model_one_to_other(y_train, X_train)

acc = 0

for i in [11,12,13]:
    X_test = scipy.io.loadmat('test/%d.mat'%(i))['de_feature']
    y_test = scipy.io.loadmat('test/%d.mat'%(i))['label']
    
    X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis = 0) - np.min(X_test, axis=0))
    acc += (one_to_other(model_list, y_test, X_test) / 3)
    
print(acc)