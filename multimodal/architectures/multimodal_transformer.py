import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet



class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, key, value):
        # Cross-attention: query attends to key and value
        attn_output, _ = self.attention(query, key, value)
        return attn_output


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
class EfficientNetEncoder(nn.Module):
    def __init__(self, output_dim=128, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNetEncoder, self).__init__()

        # Configuration for EfficientNet-B0
        base_channels = 32
        base_layers = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
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
            #c = c * 4
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
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        x = self.swish(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.swish(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=248):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding

# Load the pretrained EfficientNetB0 model
class EfficientNet_pretrained_encoder(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_pretrained_encoder, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        self.output_dim = nn.Linear(1280, 120)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        x = self.output_dim(x)

        return x


class MultimodalEncoder(nn.Module):
    def __init__(self, seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0):
        super(MultimodalEncoder, self).__init__()

        # Encoders
        # self.encoder1 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
        #                                batch_first=True, activation=nn.GELU()),
        #     num_layers=num_encoder_layers
        # )

        self.encoder1 = CrossAttention(d_model, nhead)

        self.efficient_net = EfficientNet_pretrained_encoder()
        print(f'The efficient_net has {sum(p.numel() for p in self.efficient_net.parameters() if p.requires_grad):,} trainable parameters')

        # Linear layers to adapt dimensions
        self.input_adapter1 = nn.Linear(feature_dim, d_model)
        self.image_adapter = nn.Linear(1, d_model)
        self.learned_encoding_tabular = LearnedPositionalEncoding(d_model, max_len=feature_dim)
        self.learned_encoding_images = LearnedPositionalEncoding(d_model, max_len=120)

        # Output layer
        self.output_linear = nn.Linear(d_model * 120, output_dim)  # Adjust output layer size

    def forward(self, src1, img):

        image_embeddings = self.efficient_net(img)

        src1 = self.input_adapter1(src1.unsqueeze(dim=2))  # N, 60, 25 -> N, 60, d_model
        src1 = self.learned_encoding_tabular(src1)

        image_embeddings = self.image_adapter(image_embeddings.unsqueeze(dim=2))
        image_embeddings = self.learned_encoding_images(image_embeddings)

        # Encode with transformers
        encoded1 = self.encoder1(src1, src1, image_embeddings).flatten(start_dim=1)

        output = self.output_linear(encoded1)
        output = torch.sigmoid(output)

        return output, encoded1, image_embeddings


