from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else torch.device('cpu'))
# DEVICE = 'duda'

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=32, learning_rate=0.0002, interval=100, optimizer='adam', 
                num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.G = Generator().to(DEVICE)
        self.C1 = Classifier().to(DEVICE)
        self.C2 = Classifier().to(DEVICE)
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            
            
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)
            
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
    
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, source_loader, target_loader, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for (data_src, data_tar) in zip(enumerate(source_loader), enumerate(target_loader)):
            batch_idx, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)

            self.reset_grad()
            feat_s = self.G(x_src)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, y_src)
            loss_s2 = criterion(output_s2, y_src)
            
            # train extractor and classifier
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            
            self.reset_grad()

            feat_s = self.G(x_src)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            
            feat_t = self.G(x_tar)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)
            

            loss_s1 = criterion(output_s1, y_src)
            loss_s2 = criterion(output_s2, y_src)
            
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)

            # maximize the discrepancy in target space
            loss = loss_s - loss_dis
            # loss = - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            # minimize the discrepancy in target space
            for i in range(self.num_k):
                feat_t = self.G(x_tar)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()


    def test(self, epoch, target_loader, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        rtv = []
        for batch_idx, data in enumerate(target_loader):
            (x_tar, y_tar) = data
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            feat = self.G(x_tar)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, y_tar).item()
            test_loss += F.nll_loss(output2, y_tar).item()
            output_ensemble = output1 + output2
            pred_ensemble = output_ensemble.data.max(1)[1].squeeze()
            rtv += pred_ensemble.tolist()
        return rtv
