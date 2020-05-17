import DaNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import data_loader
import mmd
import scipy.io
import json


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.02
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
DROPOUT = 0.5
N_EPOCH = 200
BATCH_SIZE = [64, 64]
LAMBDA = 0.5
GAMMA = 10 ^ 3
RESULT_TRAIN = []
RESULT_TEST = []
log_train = open('log_train_a-w.txt', 'w')
log_test = open('log_test_a-w.txt', 'w')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default=0)
parser.add_argument("--person", type=int, default=1)
args = parser.parse_args()


def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    for batch_id, (data, target) in enumerate(data_src):
        _, (x_tar, y_target) = list_tar[batch_j]
        data, target = data.to(DEVICE), target.to(DEVICE)
        x_tar, y_target = x_tar.to(DEVICE), y_target.to(DEVICE)
        model.train()
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)

        loss_c = criterion(y_src, target)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = loss_c + LAMBDA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data
        )
        batch_j += 1
        if batch_j >= len(list_tar):
            batch_j = 0
    total_loss_train /= len(data_src)
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc
    )
    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):
            data, target = data.to(DEVICE),target.to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    log_test.write(res + '\n')
    return accuracy / 100.


def dataset_load(batch_size = 64, person = args.person):
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

    X_source = (X_source - np.min(X_source, axis=0)) / (np.max(X_source, axis=0) - np.min(X_source, axis=0))
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).long().squeeze()
    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

    X_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_target = (X_target - np.min(X_target, axis=0)) / (np.max(X_target, axis=0) - np.min(X_target, axis=0))
    X_target = torch.from_numpy(X_target).float()
    y_target = torch.from_numpy(y_target).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)

    return source_dataset, target_dataset


if __name__ == '__main__':
    torch.manual_seed(args.seed)

    source_dataset, target_dataset = dataset_load(person=args.person)
    data_src = torch.utils.data.DataLoader(dataset=source_dataset,batch_size=64,shuffle=True,num_workers=1, drop_last = True)
    data_tar = torch.utils.data.DataLoader(dataset=target_dataset,batch_size=64,shuffle=True,num_workers=1, drop_last = True)

    model = DaNN.DaNN(n_input=310, n_hidden=512, n_class=4)
    model = model.to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
    )
    acc_list = []
    for e in tqdm(range(1, N_EPOCH + 1)):
        model = train(model=model, optimizer=optimizer,
                      epoch=e, data_src=data_src, data_tar=data_tar)
        acc = test(model, data_tar, e)
        acc_list.append(acc.item())
    jd = {"test_acc": acc_list}
    with open(str(args.seed)+'/acc'+str(args.person)+'.json', 'w') as f:
        json.dump(jd, f)
    
    torch.save(model, 'model_dann.pkl')
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_a-w.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_a-w.csv', res_test, fmt='%.6f', delimiter=',')