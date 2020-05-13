from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import sys
import trainer
from dataloader import datasetLoad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--outf', default='results', help='folder to output model checkpoints')
    parser.add_argument('--adv_weight', type=float, default = 0.8, help='weight for adv loss')
    parser.add_argument('--lrd', type=float, default=0.0005, help='learning rate decay, default=0.0002')
    parser.add_argument('--alpha', type=float, default = 0.5, help='multiplicative factor for target adv. loss')

    opt = parser.parse_args()
    print(opt)

    # Creating log directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass


    # Setting random seed
    random.seed(0)
    torch.manual_seed(0)
    if opt.gpu>=0:
        torch.cuda.manual_seed_all(0)

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    source_dataset, sourceval_dataset, target_dataset = datasetLoad(person=2)

    nclasses = 4
    
    # Training
    GTA_trainer = trainer.GTA(opt, nclasses, source_dataset, sourceval_dataset, target_dataset)
    GTA_trainer.train()

if __name__ == '__main__':
    main()

