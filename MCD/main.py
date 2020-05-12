import argparse
import torch
from solver import Solver
import os
import numpy as np
import scipy.io
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=120, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=3, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--person', type=int, default=2, metavar='N',
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
print(args)

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
    solver = Solver(args, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)

    record_num = 0

    record_train = 'record/k_%s_onestep_%s_%s.txt' % (args.num_k, args.one_step, record_num)
    record_test = 'record/k_%s_onestep_%s_%s_test.txt' % (args.num_k, args.one_step, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/k_%s_onestep_%s_%s.txt' % (args.num_k, args.one_step, record_num)
        record_test = 'record/k_%s_onestep_%s_%s_test.txt' % (args.num_k, args.one_step, record_num)
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')

    source_dataset, target_dataset = dataset_load(person=args.person)

    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            if not args.one_step:
                source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                num = solver.train(t, source_loader, target_loader, record_file=record_train)
            else:
                pass
                #num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                solver.test(t, target_loader, record_file=record_test, save_model=args.save_model)
            if count >= 20000:
                break

if __name__ == '__main__':
    main()