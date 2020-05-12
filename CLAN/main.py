import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import scipy.io
from solver import Solver
from torch.utils.data import DataLoader

from models import *

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CLAN network")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of samples sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--max-epoches", type=int, default=200,
                        help="Max epoches")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--num-steps", type=int, default=2800,
                        help="Number of training steps.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--snapshot-dir", type=str, default='snapshots/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--person", type=int, default=3,
                        help="choose the testing person.")
    return parser.parse_args()


args = get_arguments()

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

def main():
    source_dataset, target_dataset = dataset_load(args.person)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    step = 0

    solver = Solver(args)

    record_num = 0

    record_train = 'record/%s.txt' % (record_num)
    record_test = 'record/%s_test.txt' % (record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/_%s.txt' % (record_num)
        record_test = 'record/%s_test.txt' % (record_num)

    for _ in range(args.max_epoches):
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        solver.train(step, source_loader, target_loader)
        solver.test(target_loader, record_file = record_test)

        step += min(len(source_loader), len(target_loader))

if __name__ == '__main__':
    main()


