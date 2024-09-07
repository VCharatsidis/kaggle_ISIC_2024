import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish Activation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation Module
class SqueezeExcitation(nn.Module):
    def __init__(self, input_dim, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_dim = input_dim // squeeze_factor
        self.fc1 = nn.Conv2d(input_dim, squeeze_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_dim, input_dim, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# MBConv Block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio=4, drop_connect_rate=0.2):
        super(MBConv, self).__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.residual = (in_channels == out_channels and stride == 1)
        mid_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)
        else:
            self.expand = None

        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.se = SqueezeExcitation(mid_channels, reduction_ratio)
        self.project = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob + torch.rand((batch_size, 1, 1, 1), dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output

    def forward(self, x):
        residual = x
        if self.expand:
            x = self.swish(self.bn0(self.expand(x)))
        x = self.swish(self.bn1(self.depthwise(x)))
        x = self.se(x)
        x = self.bn2(self.project(x))
        if self.residual:
            x = self.drop_connect(x) + residual
        return x

# EfficientNet-B0 Architecture
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, drop_connect_rate=0.2):
        super(EfficientNetB0, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.swish = Swish()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        # Configuration for EfficientNet-B0
        self.blocks = nn.Sequential(
            self._make_layer(32, 16, 3, 1, 1, 1, 0),
            self._make_layer(16, 24, 3, 2, 6, 2, drop_connect_rate),
            self._make_layer(24, 40, 5, 2, 6, 2, drop_connect_rate),
            self._make_layer(40, 80, 3, 2, 6, 3, drop_connect_rate),
            self._make_layer(80, 112, 5, 1, 6, 3, drop_connect_rate),
            self._make_layer(112, 192, 5, 2, 6, 4, drop_connect_rate),
            self._make_layer(192, 320, 3, 1, 6, 1, drop_connect_rate)
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, expand_ratio, num_blocks, drop_connect_rate):
        layers = []
        layers.append(MBConv(in_channels, out_channels, kernel_size, stride, expand_ratio, drop_connect_rate=drop_connect_rate))
        for _ in range(1, num_blocks):
            layers.append(MBConv(out_channels, out_channels, kernel_size, 1, expand_ratio, drop_connect_rate=drop_connect_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# Example usage
model = EfficientNetB0(num_classes=1)
print(model)

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

# Dummy input for testing
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)
