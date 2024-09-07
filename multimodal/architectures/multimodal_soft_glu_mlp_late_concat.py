import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class just_GELU(nn.Module):
    def __init__(self, input_dim):
        super(just_GELU, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear(x)
        result = F.gelu(x)
        result = self.dropout(result)

        return result


class GLU(nn.Module):
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear(x)
        a, b = x.chunk(2, dim=-1)
        result = a + a * torch.softmax(b, dim=1)
        result = self.dropout(result)

        return result


class Tab_Attention(nn.Module):
    def __init__(self, input_dim):
        super(Tab_Attention, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * 3)
        self.dropout = nn.Dropout(0.3)
        self.input_dim = input_dim

    def forward(self, x):
        x = self.linear(x)
        V, Q, K = x.chunk(3, dim=-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        output = torch.matmul(attention_weights, V)

        output = self.dropout(output)

        return output


class VQK(nn.Module):
    def __init__(self, input_dim):
        super(VQK, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * 3)
        self.dropout = nn.Dropout(0.3)
        self.input_dim = input_dim

    def forward(self, x):
        lin_x = self.linear(x)
        V, Q, K = lin_x.chunk(3, dim=-1)

        scores = (Q * K) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        result = attention_weights * V
        result = self.dropout(result)
        result = result + V

        return result


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


class Multimodal_Soft_GLUMLP_late_concat(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=3):
        super(Multimodal_Soft_GLUMLP_late_concat, self).__init__()

        first_layer_embeddins = 2
        # First linear layer
        self.first_layer = nn.Linear(1, first_layer_embeddins)

        glu_dim = hidden_dim * first_layer_embeddins

        # GLU layers
        self.glu_layers = nn.ModuleList([just_GELU(glu_dim) for _ in range(num_layers)])

        self.efficient_net = EfficientNet_pretrained_encoder()

        # Last linear layer
        self.last_layer = nn.Linear(hidden_dim * first_layer_embeddins + 120, output_dim)
        self.dropout = nn.Dropout(0.3)

        self.loss_layer = nn.Linear(hidden_dim * first_layer_embeddins + 120, output_dim)

    def forward(self, x, img):
        image_embeddings = self.efficient_net(img)

        # Apply the first linear layer

        x = self.first_layer(x.unsqueeze(dim=2)).flatten(start_dim=1)
        x = F.relu(x)  # You can use other activations if needed

        # Apply the GLU layers
        for glu in self.glu_layers:
            x = glu(x)

        total_embeddings = x
        x = torch.cat([x, image_embeddings], dim=1)
        x = self.dropout(x)
        # Apply the last linear layer
        logit = self.last_layer(x)
        prob = torch.sigmoid(logit)

        pred_loss = self.loss_layer(x)

        return prob, pred_loss, image_embeddings, total_embeddings

# # Example usage
# input_dim = 100
# hidden_dim = 50
# output_dim = 10
# num_layers = 3
#
# model = GLUMLP(input_dim, hidden_dim, output_dim, num_layers)
# print(model)
#
# # Dummy input for testing
# input_tensor = torch.randn(32, input_dim)  # Batch size of 32
# output = model(input_tensor)
# print(output.shape)
