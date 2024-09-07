import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNet_pretrained(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_pretrained, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)

        return x


class EfficientNet_pretrained_linear(nn.Module):
    def __init__(self, out_dim, num_classes=1):
        super(EfficientNet_pretrained_linear, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        self.output_dim = nn.Linear(1280, out_dim)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        x = self.output_dim(x)

        return x


class GLU(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim * 2)

    def forward(self, x):
        x = self.linear(x)
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

class EfficientNet_regression(nn.Module):
    def __init__(self, out_dim, intermediate_dim, num_classes=1):
        super(EfficientNet_regression, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        self.output_dim = nn.Linear(intermediate_dim, out_dim)
        self.itermediate = GLU(1280, intermediate_dim)
        self.batch_norm = nn.BatchNorm1d(intermediate_dim)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        out = x.flatten(start_dim=1)

        interm = self.itermediate(out)
        interm = self.batch_norm(interm)

        x = self.output_dim(interm)

        return x, interm, out


