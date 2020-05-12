import os

import numpy as np
import scipy.io
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from models.model import *  # The model construction
from models.DomainClassifierTarget import DClassifierForTarget
from models.DomainClassifierSource import DClassifierForSource
from models.EntropyMinimizationPrinciple import EMLossForTarget
import torch
import os
import math
import torch.nn.functional as F
import argparse
import tqdm
import random

DEVICE = 'cpu'
num_classes = 4

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def opts():
    parser = argparse.ArgumentParser(description='Train alexnet on the cub200 dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=400, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b-s', type=int, default=64, help='Batch size of the source data.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='The Learning Rate.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='model saving dir')
    args = parser.parse_args()

    return args

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

def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    lr = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    global args, best_prec1
    current_epoch = 0
    args = opts()

    criterion_classifier_target = DClassifierForTarget()
    criterion_classifier_source = DClassifierForSource()
    criterion_em_target = EMLossForTarget()
    criterion = nn.CrossEntropyLoss()

    source_dataset, target_dataset = dataset_load(person = 3)
    
    record_num = 0
    record_train = 'record/%s.txt' % (record_num)
    record_test = 'record/%s_test.txt' % (record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/%s.txt' % (record_num)
        record_test = 'record/%s_test.txt' % (record_num)
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('record'):
        os.mkdir('record')

    G = Generator()
    C = Classifier()

    opt_g = optim.Adam(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_c = optim.Adam(C.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        adjust_learning_rate(opt_g, epoch, args)
        adjust_learning_rate(opt_c, epoch, args)
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        len_dataloader = min(len(source_loader), len(target_loader))
        LOSS_G = 0
        LOSS_C = 0

        for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(source_loader), enumerate(target_loader)), total=len_dataloader, leave=False):
            batch_idx, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            y_src_temp = y_src + num_classes
            adjust_learning_rate(opt_g, epoch, args)
            adjust_learning_rate(opt_c, epoch, args)

            opt_g.zero_grad()
            opt_c.zero_grad()
            output_source = C(G(x_src))
            output_target = C(G(x_tar))
            
            loss_task_s_Cs = criterion(output_source[:,:num_classes], y_src)
            loss_task_s_Ct = criterion(output_source[:,num_classes:], y_src)
            loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
            loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)
            loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2   ### used to classifier
            LOSS_C += loss_classifier.item() / len_dataloader
            loss_classifier.backward()
            opt_c.step()

            opt_g.zero_grad()
            opt_c.zero_grad()
            output_source = C(G(x_src))
            output_target = C(G(x_tar))
            loss_category_st_G = 0.5 * criterion(output_source, y_src) + 0.5 * criterion(output_source, y_src_temp)
            loss_domain_st_G = 0.5 * criterion_classifier_target(output_target) + 0.5 * criterion_classifier_source(output_target)
            loss_target_em = criterion_em_target(output_target)
            lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
            loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em)  ### used to feature extractor
            LOSS_G += loss_G.item() / len_dataloader
            loss_G.backward()
            opt_g.step()

        print('Train Epoch: {} \tLoss_G: {:.6f}\t Loss_C: {:.6f}'.format(epoch, LOSS_G, LOSS_C))
        record = open(record_test, 'a')
        record.write('Train Epoch: {} \tLoss_G: {:.6f}\t Loss_C: {:.6f}'.format(epoch, LOSS_G, LOSS_C))
        record.close()
        
        # Begin test 
        G.eval()
        C.eval()
        
        correct1 = 0
        correct2 = 0
        size = 0

        for batch_idx, data in enumerate(target_loader):
            (x_tar, y_tar) = data
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            output = C(G(x_tar))
            out1 = output[:, :num_classes]
            out2 = output[:, num_classes:]
            pred1 = out1.data.max(1)[1]
            pred2 = out2.data.max(1)[1]
            correct1 += pred1.eq(y_tar.data).cpu().sum()
            correct2 += pred2.eq(y_tar.data).cpu().sum()
            k = y_tar.data.size()[0]
            size += k
        
        print('\nTraning epoch: {}\t Test set: Accuracy C1: {}/{} ({:.2f}%), Accuracy C2: {}/{} ({:.2f}%)\n'.format(epoch,
                                        correct1, size, 100. * correct1 / size,
                                        correct2, size, 100. * correct2 / size))
        record = open(record_test, 'a')
        record.write('\nTraning epoch: {}\t Test set: Accuracy C1: {}/{} ({:.2f}%), Accuracy C2: {}/{} ({:.2f}%)\n'.format(epoch,
                                        correct1, size, 100. * correct1 / size,
                                        correct2, size, 100. * correct2 / size))
        record.close()

        torch.save(G,'%s/model_epoch%s_G.pt' % (args.checkpoint_dir, epoch))
        torch.save(C,'%s/model_epoch%s_C.pt' % (args.checkpoint_dir, epoch))

if __name__ == '__main__':
    main()




