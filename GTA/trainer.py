import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils
from torch.utils.data import DataLoader
import tqdm

class GTA(object):
    def __init__(self, opt, nclasses, source_dataset, sourceval_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.sourceval_dataset = sourceval_dataset
        self.target_dataset = target_dataset
        self.opt = opt
        self.best_val = 0
        
        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netG = models._netG(opt, nclasses)
        self.netD = models._netD(opt, nclasses)
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses)

        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu>=0:
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Other variables
        self.real_label_val = 0
        self.fake_label_val = 1

    """
    Validation function
    """
    def validate(self, epoch):
        
        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0
    
        # Testing the model
        source_valloader = DataLoader(self.sourceval_dataset, batch_size=self.opt.batchSize, shuffle=False, num_workers=1)
        for i, datas in enumerate(source_valloader):
            inputs, labels = datas
            
            if self.opt.gpu >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputv, labelv = Variable(inputs, volatile=True), Variable(labels) 

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels).sum())
            
        val_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))
    
        # Saving checkpoints
        torch.save(self.netF.state_dict(), '%s/models/netF.pth' %(self.opt.outf))
        torch.save(self.netC.state_dict(), '%s/models/netC.pth' %(self.opt.outf))
        
        if val_acc>self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/model_best_netF.pth' %(self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/model_best_netC.pth' %(self.opt.outf))

        # Testing
    
        self.netF.eval()
        self.netC.eval()
            
        total = 0
        correct = 0
        targetloader = DataLoader(self.target_dataset, batch_size=self.opt.batchSize, shuffle=False, num_workers=1)
        for i, datas in enumerate(targetloader):
            inputs, labels = datas
            if self.opt.gpu>=0:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                inputv, labelv = Variable(inputs), Variable(labels)

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)        
            total += labels.size(0)
            correct += ((predicted == labels).sum())
            
        test_acc = 100*float(correct)/total
        print('Test Accuracy: %f %%' % (test_acc))
            
            
    """
    Train function
    """
    def train(self):
        
        curr_iter = 0
        
        for epoch in range(self.opt.nepochs):
            source_trainloader = DataLoader(self.source_dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=1)
            targetloader = DataLoader(self.target_dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=1)
            self.netG.train()    
            self.netF.train()    
            self.netC.train()    
            self.netD.train()    

            len_dataloader = min(len(source_trainloader), len(targetloader))

            for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(source_trainloader), enumerate(targetloader)), total=len_dataloader, leave=False):
                self.real_label_val = 0.9 + torch.rand(1).item() * 0.1
                self.fake_label_val = 0 + torch.rand(1).item() * 0.1
                ###########################
                # Forming input variables
                ###########################

                batch_idx, (src_inputs, src_labels) = data_src
                _, (tgt_inputs, _) = data_tar    
                #src_inputs_unnorm = (((src_inputs*self.std[0]) + self.mean[0]) - 0.5)*2

                if src_inputs.shape[0] != tgt_inputs.shape[0]:
                    continue
                # Creating one hot vector
                size = src_inputs.shape[0]
                labels_onehot = np.zeros((size, self.nclasses+1), dtype=np.float32)
                for num in range(size):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((size, self.nclasses+1), dtype=np.float32)
                for num in range(size):
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                reallabel = torch.FloatTensor(size).fill_(self.real_label_val)
                fakelabel = torch.FloatTensor(size).fill_(self.fake_label_val)
                if self.opt.gpu>=0:
                    reallabel, fakelabel = reallabel.cuda(), fakelabel.cuda()
                reallabelv = Variable(reallabel) 
                fakelabelv = Variable(fakelabel) 
                
                if self.opt.gpu>=0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    #src_inputs_unnorm = src_inputs_unnorm.cuda() 
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()
                
                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                #src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)
                
                ###########################
                # Updates
                ###########################
                
                # Updating D network
                
                self.netD.zero_grad()
                src_emb = self.netF(src_inputsv)
                src_emb_cat = torch.cat((src_labels_onehotv, src_emb), 1)
                src_gen = self.netG(src_emb_cat)

                tgt_emb = self.netF(tgt_inputsv)
                tgt_emb_cat = torch.cat((tgt_labels_onehotv, tgt_emb),1)
                tgt_gen = self.netG(tgt_emb_cat)

                # 真实源域图片用来辨别真实并且分类
                src_realoutputD_s, src_realoutputD_c = self.netD(src_inputsv)   
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv) 
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv) 

                # 生成的源域图片用来辨别虚假
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                # 生成的目标域图片用来辨别虚假
                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)          
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)    
                self.optimizerD.step()
                
                self.netG.zero_grad()       
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                # G让生成的图像仍然可以被正确地分类
                errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                # G的目标是要让生成的图像被认为是真实的
                errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()
                for i in range(5):
                    # Updating G network
                    self.netG.zero_grad()       
                    src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                    # G让生成的图像仍然可以被正确地分类
                    errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                    # G的目标是要让生成的图像被认为是真实的
                    errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                    errG = errG_c + errG_s
                    errG.backward(retain_graph=True)
                    self.optimizerG.step()
                

                # Updating C network
                # C使Embedding更好地被分类
                self.netC.zero_grad()
                outC = self.netC(src_emb)   
                errC = self.criterion_c(outC, src_labelsv)
                errC.backward(retain_graph=True)    
                self.optimizerC.step()

                
                # Updating F network
                self.netF.zero_grad()
                # 这个和上面的errC是一样的，但是因为前面的时候F已经zero_grad，所以再传一次
                errF_fromC = self.criterion_c(outC, src_labelsv)        
                # 让生成的source能被分类得更准确
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv)*(self.opt.adv_weight)

                # 让生成的target被认为是真实的图像 （但是为什么不加上让生成的source被认为真实这一项呢？）
                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(self.opt.adv_weight*self.opt.alpha)
                
                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()        
                
                curr_iter += 1
                
                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd, curr_iter)    
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd, curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd, curr_iter)                  

                print('ErrD: %.2f, ErrG: %.2f, ErrC: %.2f, ErrF: %.2f' % (errD.item(), errG.item(), errC.item(), errF.item()))

            # Validate every epoch
            self.validate(epoch+1)