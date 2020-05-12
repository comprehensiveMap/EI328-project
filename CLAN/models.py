import torch
import torch.nn as nn
import torch.nn.functional as F

EXTRACTED_FEATURE_DIM = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_first_linear', nn.Linear(310, 256))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_second_linear', nn.Linear(256, EXTRACTED_FEATURE_DIM))
        feature.add_module('f_drop1', nn.Dropout(0.2))
        feature.add_module('f_relu2', nn.ReLU(True)) # 直接覆盖原有内存，节约空间和时间
        self.feature = feature

    def forward(self, x): # x: (batch_size, ori_feature_dim(310))
        return self.feature(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(EXTRACTED_FEATURE_DIM, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.3))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, 4))

    def forward(self, x):
        return self.class_classifier(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        #x = torch.sigmoid(x)

        return x

'''
class Discriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        #x = torch.sigmoid(x)

        return x
'''

class WeightedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
                
        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)

