import torch
import torch.nn as nn
import math


EXTRACTED_FEATURE_DIM = 128

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
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
    def __init__(self, class_num=4, extract=True, dropout_p=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(EXTRACTED_FEATURE_DIM, 100),
            nn.BatchNorm1d(100, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc2 = nn.Linear(100, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit
        return logit
