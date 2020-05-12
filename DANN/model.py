import torch
import torch.nn as nn
import adv_layer


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


class DANN(nn.Module):

    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.domain_classifier = adv_layer.Discriminator(
            input_dim=EXTRACTED_FEATURE_DIM, hidden_dim=100)

    def forward(self, input_data, alpha=1, source=True): # input_data: (batch_size, ori_dim(310))
        feature = self.feature(input_data) # (batch_size, EXTRACTED_FEATURE_DIM)
        class_output = self.classifier(feature) # (batch_size, 4)
        domain_output = self.get_adversarial_result(
            feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1): # x: (batch_size, EXTRACTED_FEATURE_DIM)
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device) # (batch_size, )
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = adv_layer.ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv
