from torch import nn

EXTRACTED_FEATURE_DIM = 128

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
