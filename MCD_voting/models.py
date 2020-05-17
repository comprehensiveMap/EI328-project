import torch
import torch.nn as nn

EXTRACTED_FEATURE_DIM = 128
HIDDEN_DIM = 256

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_first_linear', nn.Linear(310, HIDDEN_DIM))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_second_linear', nn.Linear(HIDDEN_DIM, EXTRACTED_FEATURE_DIM))
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


