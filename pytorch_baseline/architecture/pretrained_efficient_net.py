import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# Load the pretrained EfficientNetB0 model
class EfficientNetWithFeatures(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetWithFeatures, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        embeddings = x
        x = self.dropout(x)
        x = self.base_model._fc(x)
        x = torch.sigmoid(x)
        return x, embeddings


