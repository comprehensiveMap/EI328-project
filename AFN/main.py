import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import random
import scipy.io
from torch.utils.data import DataLoader
from models import *
import json


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--pre_epochs", default=50, type=int)
parser.add_argument("--epoch", default=300, type=int)
parser.add_argument("--snapshot", default="snapshot")
parser.add_argument("--lr", default=0.0002)
parser.add_argument("--class_num", default=4)
parser.add_argument("--extract", default=True)
parser.add_argument("--radius", default=9)
parser.add_argument("--weight_L2norm", default=0.005)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--seed", type = int, default=0)
parser.add_argument("--person", type = int, default=1)
args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def dataset_load(batch_size = 64, person = 1):
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

DEVICE = 'cuda'

setup_seed(args.seed)
source_dataset, target_dataset = dataset_load(person = args.person)

netG = FeatureExtractor().to(DEVICE)
netF = Classifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).to(DEVICE)

record_num = 0
record_test = 'record/%s_test.txt' % (record_num)
while os.path.exists(record_test):
    record_num += 1
    record_test = 'record/%s_test.txt' % (record_num)

if not os.path.exists('snapshot'):
    os.mkdir('snapshot')

if not os.path.exists('record'):
    os.mkdir('record')

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - args.radius) ** 2
    return args.weight_L2norm * l

opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

#opt_g = optim.Adam(netG.parameters(), lr=args.lr, weight_decay=0.0005)
#opt_f = optim.Adam(netF.parameters(), lr=args.lr, weight_decay=0.0005)


for epoch in range(1, args.pre_epochs + 1):
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    LOSS = 0
    for i, (x_src, y_src) in tqdm.tqdm(enumerate(source_loader)):
        x_src = Variable(x_src.to(DEVICE))
        y_src = Variable(y_src.to(DEVICE))

        opt_g.zero_grad()
        opt_f.zero_grad()

        s_bottleneck = netG(x_src)
        s_fc2_emb, s_logit = netF(s_bottleneck)

        s_fc2_ring_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        s_cls_loss = get_cls_loss(s_logit, y_src)

        loss = s_cls_loss + s_fc2_ring_loss
        LOSS += loss.item() / len(source_loader)

        loss.backward()

        opt_g.step()
        opt_f.step()

    print('Train Epoch: {} \t Pretrain Loss: {:.6f}\t'.format(epoch, LOSS))

acc_list = []
for epoch in range(1, args.epoch + 1):
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    len_dataloader = min(len(source_loader), len(target_loader))

    LOSS = 0
    for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(source_loader), enumerate(target_loader)), total=len_dataloader, leave=False):
        batch_idx, (x_src, y_src) = data_src
        _, (x_tar, _) = data_tar

        x_src = Variable(x_src.to(DEVICE))
        y_src = Variable(y_src.to(DEVICE))
        x_tar = Variable(x_tar.to(DEVICE))

        opt_g.zero_grad()
        opt_f.zero_grad()

        s_bottleneck = netG(x_src)
        t_bottleneck = netG(x_tar)
        s_fc2_emb, s_logit = netF(s_bottleneck)
        t_fc2_emb, t_logit = netF(t_bottleneck)

        s_cls_loss = get_cls_loss(s_logit, y_src)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

        loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        LOSS += loss.item() / len_dataloader
        loss.backward()

        opt_g.step()
        opt_f.step()
    
    print('Train Epoch: {} \tLoss: {:.6f}\t'.format(epoch, LOSS))
    torch.save(netG.state_dict(), os.path.join(args.snapshot, "_netG_" + str(epoch) + ".pth"))
    torch.save(netF.state_dict(), os.path.join(args.snapshot, "_netF_" + str(epoch) + ".pth"))

    # Begin test 
    netG.eval()
    netF.eval()
    
    correct = 0
    size = 0

    for batch_idx, data in enumerate(target_loader):
        (x_tar, y_tar) = data
        x_tar = Variable(x_tar.to(DEVICE))
        _, pred = netF(netG(x_tar))
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        y_tar = y_tar.numpy()
        correct += np.equal(y_tar, pred).sum()
        k = y_tar.shape[0]
        size += k

    acc = correct * 100.0 / size
    acc_list.append(acc/100.)
    print("Epoch {0}: {1}".format(epoch, correct))
    
    print('\nTraning epoch: {}\t Test set: Accuracy C: {:.2f}%'.format(epoch, acc))
    record = open(record_test, 'a')
    record.write('\nTraning epoch: {}\t Test set: Accuracy C: {:.2f}%'.format(epoch, acc))
    record.close()

jd = {"test_acc": acc_list}
with open(str(args.seed)+'/acc'+str(args.person)+'.json', 'w') as f:
    json.dump(jd, f)


