import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# MBConv Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, reduction_ratio=4, kernel_size=3):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio

        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(hidden_dim)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-Excitation phase
        self.se = SEBlock(hidden_dim, reduction_ratio)

        # Projection phase
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        residual = x
        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + residual
        return x


# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # Configuration for EfficientNet-B0
        base_channels = 32
        base_layers = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [2, 24, 2, 2, 3],
            [2, 40, 2, 2, 5],
            [2, 80, 3, 2, 3],
            [2, 112, 3, 1, 5],
            [2, 192, 4, 2, 5],
            [2, 320, 1, 1, 3],
        ]

        # Initial convolution layer
        out_channels = int(base_channels * width_coefficient)
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU()

        # Build MBConv blocks
        layers = []
        in_channels = out_channels
        for t, c, n, s, k in base_layers:
            c = c * 4
            out_channels = int(c * width_coefficient)
            repeats = int(n * depth_coefficient)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_channels, t, stride, kernel_size=k))
                in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        # Final layers
        out_channels = int(1280 * width_coefficient)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        x = self.swish(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.swish(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# # Example usage
# model = EfficientNet()
# print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
# input = torch.randn(1, 3, 384, 384)
# output = model(input)
# print(output.shape)
