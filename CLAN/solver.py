from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
import tqdm
import os
import os.path as osp

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
DEVICE = 'cpu'
EPSILON = 0.4
LAMBDA_LOCAL = 5
LAMBDA_ADV = 0.001
LAMBDA_WEIGHT = 0.01

# Training settings
class Solver(object):
    def __init__(self, args):
        self.num_steps = args.num_steps
        self.power = args.power
        self.lr = args.learning_rate
        self.lr_D = args.learning_rate_D
        self.batch_size = args.batch_size
        self.snapshot_dir = args.snapshot_dir
        self.FE = Generator()
        self.C1 = Classifier()
        self.C2 = Classifier()
        self.D = Discriminator()
        self.preheat_steps = int(self.num_steps / 20)
        self.set_optimizer()

    def set_optimizer(self):
        self.opt_fe = optim.Adam(self.FE.parameters(),
                                lr=self.lr, weight_decay=0.0005)
        self.opt_c1 = optim.Adam(self.C1.parameters(),
                                    lr=self.lr, weight_decay=0.0005)
        self.opt_c2 = optim.Adam(self.C2.parameters(),
                                    lr=self.lr, weight_decay=0.0005)
        self.opt_d = optim.Adam(self.D.parameters(),
                                    lr=self.lr_D, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_fe.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_d.zero_grad()

    def adjust_learning_rate(self, step):
        if step < self.preheat_steps:
            lr = self.lr * (float(step) / self.preheat_steps)
        else:
            lr = self.lr * ((1 - float(step) / self.num_steps) ** (self.power))

        self.opt_fe.param_groups[0]['lr'] = lr
        self.opt_c1.param_groups[0]['lr'] = lr
        self.opt_c2.param_groups[0]['lr'] = lr

    def adjust_learning_rate_D(self, step):
        if step < self.preheat_steps:
            lr = self.lr_D * (float(step) / self.preheat_steps)
        else:
            lr = self.lr_D * ((1 - float(step) / self.num_steps) ** (self.power))
        
        self.opt_d.param_groups[0]['lr'] = lr

    def weightmap(self, pred1, pred2):
        output = 1.0 - torch.cosine_similarity(pred1, pred2, dim=1)
        return output

    def train(self, step, source_loader, target_loader, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        bce_loss = torch.nn.BCEWithLogitsLoss()
        weighted_bce_loss = WeightedBCEWithLogitsLoss()
        self.FE.train()
        self.C1.train()
        self.C2.train()
        self.D.train()
        torch.cuda.manual_seed(1)

        len_dataloader = min(len(source_loader), len(target_loader))

        for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(source_loader), enumerate(target_loader)), total=len_dataloader, leave=False):
            self.reset_grad()
            self.adjust_learning_rate(step=step)
            self.adjust_learning_rate_D(step=step)

            damping = (1 - step / self.num_steps)
            #damping = 1

            #======================================================================================
            # train FE
            #======================================================================================

            #Remove Grads in D
            for param in self.D.parameters():
                param.requires_grad = False

            # Train with Source
            batch_idx, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)

            fe_s = self.FE(x_src) 
            outC1_s = self.C1(fe_s)
            outC2_s = self.C2(fe_s)

            #Classification loss
            loss_clf = criterion(outC1_s,y_src) + criterion(outC2_s,y_src)
            loss_clf.backward()

            # Train with Target
            fe_t = self.FE(x_tar)
            outC1_t = self.C1(fe_t)
            outC2_t = self.C2(fe_t)

            weight_map = self.weightmap(F.softmax(outC1_t, dim=1), F.softmax(outC2_t, dim=1))

            #out_D = self.D(F.softmax(outC1_t + outC2_t, dim = 1))
            outD_t = self.D(fe_t)

            #Adaptive Adversarial Loss
            t_truth = Variable(torch.FloatTensor(outD_t.data.size()).fill_(0)).to(DEVICE)
            if(step > self.preheat_steps):
                loss_adv = weighted_bce_loss(outD_t, t_truth, weight_map, EPSILON, LAMBDA_LOCAL)
            else:
                loss_adv = bce_loss(outD_t, t_truth)

            loss_adv = loss_adv * LAMBDA_ADV * damping
            loss_adv.backward()

            
            #Weight Discrepancy Loss
            w_C1 = None
            w_C2 = None

            for (w1, w2) in zip(self.C1.parameters(), self.C2.parameters()):
                if w_C1 is None and w_C2 is None:
                    w_C1 = w1.view(-1)
                    w_C2 = w2.view(-1)
                else:
                    w_C1 = torch.cat((w_C1, w1.view(-1)), 0)
                    w_C2 = torch.cat((w_C2, w2.view(-1)), 0)

            loss_weight = (torch.matmul(w_C1, w_C2)) / (torch.norm(w_C1) * torch.norm(w_C2)) + 1 # +1 is for a positive loss
            loss_weight = loss_weight * LAMBDA_WEIGHT * damping * 2
            loss_weight.backward()
            
            #======================================================================================
            # train D
            #======================================================================================
            
            # Bring back Grads in D
            for param in self.D.parameters():
                param.requires_grad = True

            # Train with Source
            fe_s = fe_s.detach()
            outD_s = self.D(fe_s)
            loss_D_s = bce_loss(outD_s,
                          Variable(torch.FloatTensor(outD_s.data.size()).fill_(0)).to(DEVICE))
            loss_D_s.backward()

            # Train with Target
            fe_t = fe_t.detach()
            outC1_t = outC1_t.detach()
            outC2_t = outC2_t.detach()
            weight_map = weight_map.detach()
            D_out_t = self.D(fe_t)

            #Adaptive Adversarial Loss
            if(step > self.preheat_steps):
                loss_D_t = weighted_bce_loss(D_out_t, Variable(torch.FloatTensor(D_out_t.data.size()).fill_(1)).to(DEVICE), 
                                            weight_map, EPSILON, LAMBDA_LOCAL)
            else:
                loss_D_t = bce_loss(D_out_t,
                            Variable(torch.FloatTensor(D_out_t.data.size()).fill_(1)).to(DEVICE))

            loss_D_t.backward()
            
            self.opt_fe.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_d.step()

            #print('exp = {}'.format(args.snapshot_dir))
            '''
            print(
            'iter = {0:6d}/{1:6d}, loss_clf = {2:.4f} loss_adv = {3:.4f}, loss_D_s = {4:.4f} loss_D_t = {5:.4f}'.format(
                step, self.num_steps, loss_clf, loss_adv, loss_D_s, loss_D_t))

            f_loss = open(osp.join(self.snapshot_dir,'loss.txt'), 'a')
            f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f}\n'.format(
                loss_clf, loss_adv, loss_D_s, loss_D_t))
            f_loss.close()
            '''            
            
            print(
            'iter = {0:6d}/{1:6d}, loss_clf = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
                step, self.num_steps, loss_clf, loss_adv, loss_weight, loss_D_s, loss_D_t))

            f_loss = open(osp.join(self.snapshot_dir,'loss.txt'), 'a')
            f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
                loss_clf, loss_adv, loss_weight, loss_D_s, loss_D_t))
            f_loss.close()
            

            step += 1
            

        print('taking snapshot ...')
        torch.save(self.FE.state_dict(), osp.join(self.snapshot_dir, str(step) + '_FE.pth'))
        torch.save(self.C1.state_dict(), osp.join(self.snapshot_dir, str(step) + '_C1.pth'))
        torch.save(self.C2.state_dict(), osp.join(self.snapshot_dir, str(step) + '_C2.pth'))
        torch.save(self.D.state_dict(), osp.join(self.snapshot_dir, str(step) + '_D.pth'))


    def test(self, target_loader, record_file=None):
        self.FE.eval()
        self.C1.eval()
        self.C2.eval()

        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0

        for batch_idx, data in enumerate(target_loader):
            (x_tar, y_tar) = data
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            feat = self.FE(x_tar)
            output1 = self.C1(feat)
            output2 = self.C2(feat)

            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = y_tar.data.size()[0]
            correct1 += pred1.eq(y_tar.data).cpu().sum()
            correct2 += pred2.eq(y_tar.data).cpu().sum()
            correct3 += pred_ensemble.eq(y_tar.data).cpu().sum()
            size += k
        
        print(
            '\nTest set: Accuracy C1: {}/{} ({:.2f}%) Accuracy C2: {}/{} ({:.2f}%) Accuracy Ensemble: {}/{} ({:.2f}%) \n'.format(
                correct1, size, 100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        
        if record_file:
            record = open(record_file, 'w')
            print('recording %s', record_file)
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
