import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import models
import utils
import os
import argparse
from dataloader import dataset_load
from torch.utils.data import DataLoader

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--checkpoint_dir', default='results/models', help='folder to load model checkpoints from')
    parser.add_argument('--model_best', type=int, default=0, help='Flag to specify whether to use the best validation model or last checkpoint| 1-model best, 0-current checkpoint')

    opt = parser.parse_args()

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # Creating data loader

    source_dataset, sourceval_dataset, target_dataset = dataset_load(person=1)
    targetloader = DataLoader(target_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=1)

    nclasses = 4
    
    # Creating and loading models
    
    netF = models._netF(opt)
    netC = models._netC(opt, nclasses)
    
    if opt.model_best == 0: 
        netF_path = os.path.join(opt.checkpoint_dir, 'netF.pth')
        netC_path = os.path.join(opt.checkpoint_dir, 'netC.pth')
    else:
        netF_path = os.path.join(opt.checkpoint_dir, 'model_best_netF.pth')
        netC_path = os.path.join(opt.checkpoint_dir, 'model_best_netC.pth')
    
        
    netF.load_state_dict(torch.load(netF_path))
    netC.load_state_dict(torch.load(netC_path))
    
    if opt.gpu>=0:
        netF.cuda()
        netC.cuda()
        
    # Testing
    
    netF.eval()
    netC.eval()
        
    total = 0
    correct = 0

    for i, datas in enumerate(targetloader):
        inputs, labels = datas
        if opt.gpu>=0:
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            inputv, labelv = Variable(inputs), Variable(labels)

        outC = netC(netF(inputv))
        _, predicted = torch.max(outC.data, 1)        
        total += labels.size(0)
        correct += ((predicted == labels).sum())
        
    test_acc = 100*float(correct)/total
    print('Test Accuracy: %f %%' % (test_acc))


if __name__ == '__main__':
    main()

