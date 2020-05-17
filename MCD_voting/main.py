import argparse
import torch
from solver import Solver
import os
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=120, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=3, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--person', type=int, default=3, metavar='N',
                    help='the person for testing')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def dataset_load(person_source=1, person_target=args.person):
    X_source = np.array([])
    y_source = np.array([])

    data = scipy.io.loadmat('../train/%d.mat'%(person_source))['de_feature']
    label = scipy.io.loadmat('../train/%d.mat'%(person_source))['label']
    X_source = data
    y_source = label

    X_source = (X_source - np.min(X_source, axis=0)) / (np.max(X_source, axis=0) - np.min(X_source, axis=0))
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).long().squeeze()
    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

    X_target = scipy.io.loadmat('../test/%d.mat'%(10 + person_target))['de_feature']
    y_target = scipy.io.loadmat('../test/%d.mat'%(10 + person_target))['label']
    X_target = (X_target - np.min(X_target, axis=0)) / (np.max(X_target, axis=0) - np.min(X_target, axis=0))
    X_target = torch.from_numpy(X_target).float()
    y_target = torch.from_numpy(y_target).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)

    return source_dataset, target_dataset

def main():
    solvers = [Solver(args, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch) for _ in range(10)]

    datasets = [dataset_load(person_source=i) for i in range(1, 11)]
    y_label = scipy.io.loadmat('../test/%d.mat'%(10 + args.person))['label']
    y_label = torch.from_numpy(y_label).long().squeeze()

    for t in range(args.max_epoch):
        preds = []
        accs = []
        for idx, solver in tqdm(enumerate(solvers), total=len(solvers), leave=False):
            source_dataset, target_dataset = datasets[idx]
            source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
            target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
            solver.train(t, source_loader, target_loader)

            if t % 1 == 0:
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, num_workers=1)
                pred = solver.test(t, target_loader, save_model=args.save_model)
                preds.append(pred)
                pred = torch.tensor(pred, dtype=torch.long)
                tmp_acc = (y_label == pred).sum().item() / len(pred)
                accs.append(tmp_acc)
        
        voted_pred = []
        for j in range(len(y_label)):
            label_count = [0]*4
            for i in range(len(preds)):
                label_count[preds[i][j]] += 1
            max_label = label_count.index(max(label_count))
            voted_pred.append(max_label)
        voted_pred = torch.tensor(voted_pred)
        acc = (y_label == voted_pred).sum().item() / len(voted_pred)
        print("In epoch %d, voted_acc: %.4f" %(t, acc))

if __name__ == '__main__':
    main()