import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Generator network
"""
class _netG(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netG, self).__init__()
        
        self.ndim = 128
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.Linear(self.nz + self.ndim + self.nclasses + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 310),
            nn.BatchNorm1d(310),
            nn.Tanh(),
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1)
        noise = torch.FloatTensor(batchSize, self.nz).normal_(0, 1)  
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        output = self.main(torch.cat((input, noisev),1))
        return output

"""
Discriminator network
"""
class _netD(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),      
        )

        self.classifier_c = nn.Sequential(nn.Linear(256, nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(256, 1), 
        						nn.Sigmoid())              

    def forward(self, input):       
        output = self.feature(input)
        output_s = self.classifier_s(output.view(-1, 256))
        output_s = output_s.view(-1)
        output_c = self.classifier_c(output.view(-1, 256))
        return output_s, output_c

"""
Feature extraction network
"""
class _netF(nn.Module):
    def __init__(self, opt):
        super(_netF, self).__init__()
        
        feature = nn.Sequential()
        feature.add_module('f_first_linear', nn.Linear(310, 256))
        feature.add_module('f_bn', nn.BatchNorm1d(256))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_second_linear', nn.Linear(256, 128))
        feature.add_module('f_bn2', nn.BatchNorm1d(128))
        feature.add_module('f_relu2', nn.ReLU(True)) # 直接覆盖原有内存，节约空间和时间
        self.feature = feature

    def forward(self, x): # x: (batch_size, ori_feature_dim(310))
        return self.feature(x)

"""
Classifier network
"""
class _netC(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netC, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, 4))

    def forward(self, x):
        return self.class_classifier(x)

