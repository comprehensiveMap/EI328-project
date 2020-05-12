from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
import tqdm

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
DEVICE = 'cpu'

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
        self.G = Generator()
        self.C1 = Classifier()
        self.C2 = Classifier()
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

        len_dataloader = min(len(source_loader), len(target_loader))

        for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(source_loader), enumerate(target_loader)), total=len_dataloader, leave=False):
            batch_idx, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)

            self.reset_grad()
            feat_s = self.G(x_src)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, y_src)
            loss_s2 = criterion(output_s2, y_src)

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

            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(self.num_k):
                feat_t = self.G(x_tar)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.item(), loss_s2.item(), loss_dis.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s1.item(), loss_s2.item()))
                    record.close()
        return batch_idx

    def test(self, epoch, target_loader, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        for batch_idx, data in enumerate(target_loader):
            (x_tar, y_tar) = data
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            feat = self.G(x_tar)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, y_tar).item()
            test_loss += F.nll_loss(output2, y_tar).item()
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = y_tar.data.size()[0]
            correct1 += pred1.eq(y_tar.data).cpu().sum()
            correct2 += pred2.eq(y_tar.data).cpu().sum()
            correct3 += pred_ensemble.eq(y_tar.data).cpu().sum()
            size += k
        test_loss = test_loss / size / 2
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/model_epoch%s_G.pt' % (self.checkpoint_dir, epoch))
            torch.save(self.C1,
                       '%s/model_epoch%s_C1.pt' % (self.checkpoint_dir, epoch))
            torch.save(self.C2,
                       '%s/model_epoch%s_C2.pt' % (self.checkpoint_dir, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
