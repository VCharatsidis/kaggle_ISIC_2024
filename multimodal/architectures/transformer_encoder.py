import torch
import torch.nn as nn
import math


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding


class SimpleEncoder(nn.Module):
    def __init__(self, seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0):
        super(SimpleEncoder, self).__init__()

        # Encoders
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True, activation=nn.GELU()),
            num_layers=num_encoder_layers
        )

        # Linear layers to adapt dimensions
        self.input_adapter1 = nn.Linear(feature_dim, d_model)
        self.learned_encoding_1 = LearnedPositionalEncoding(d_model, max_len=feature_dim)

        # Output layer
        self.output_linear = nn.Linear(d_model * seq_length, output_dim)  # Adjust output layer size

    def forward(self, src1):
        src1 = self.input_adapter1(src1.unsqueeze(dim=2))  # N, 60, 25 -> N, 60, d_model
        src1 = self.learned_encoding_1(src1)

        # Encode with transformers
        encoded1 = self.encoder1(src1).flatten(1)

        output = self.output_linear(encoded1)
        output = torch.sigmoid(output)
        return output


