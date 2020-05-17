from torch import nn
import torch.nn.functional as F

EXTRACTED_FEATURE_DIM = 128
# EXTRACTED_FEATURE_DIM = 64


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(310, 256),
#             nn.ReLU(True),
#             nn.Linear(256, EXTRACTED_FEATURE_DIM),
#             nn.Dropout(0.2),
#             nn.ReLU(True),
#         )
        
#         self.classifier_t = nn.Sequential(
#             nn.Linear(EXTRACTED_FEATURE_DIM, 100),
#             nn.BatchNorm1d(100),
#             nn.ReLU(True),
#             nn.Dropout(0.3),
#             nn.Linear(100, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(True),
#             nn.Linear(64, 4)
#         )

#         self.residual = nn.Sequential(
#             nn.ReLU(True),
#             nn.Linear(4, 4),
#             nn.ReLU(True),
#             nn.Linear(4, 4)
#         )
        
#         self.classifier = nn.Sequential(
#             self.classifier_t(),
#             self.residual()
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256),
            nn.ReLU(True),
            nn.Linear(256, EXTRACTED_FEATURE_DIM),
            nn.Dropout(0.2),
            nn.ReLU(True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(EXTRACTED_FEATURE_DIM, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits



# class cnn_feature_extractor(nn.Module):
#     def __init__(self, out_channels=256):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, out_channels, kernel_size=[1, 62])
#         self.fc1 = nn.Sequential(nn.Linear(out_channels, EXTRACTED_FEATURE_DIM),
#                                nn.ReLU(True))
        
#     def forward(self, datas):
#         datas = datas.view(-1, 5, 62) # (batch_size, 5, 62)
#         expand_datas = datas.unsqueeze(1) # (batch_size, 1, 5, 62)
#         conved = self.conv1(expand_datas).squeeze() # (batch_size, out_channels, 5)
#         pooled = F.max_pool1d(conved, kernel_size=5).squeeze() # (batch_size, out_channels)
#         out = self.fc1(pooled) # (batch_size, EXTRACTED_FEATURE_DIM)
#         return out


# class Net(nn.Module):
#     def __init__(self, out_channels=256):
#         super().__init__()
#         self.feature_extractor = cnn_feature_extractor()
        
#         self.classifier = nn.Sequential(
#             nn.Linear(EXTRACTED_FEATURE_DIM, 100),
#             nn.BatchNorm1d(100),
#             nn.ReLU(True),
#             nn.Dropout(0.3),
#             nn.Linear(100, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(True),
#             nn.Linear(64, 4)
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits
