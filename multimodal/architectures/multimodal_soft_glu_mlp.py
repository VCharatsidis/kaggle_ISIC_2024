import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class just_GELU(nn.Module):
    def __init__(self, input_dim):
        super(just_GELU, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear(x)
        result = F.gelu(x)
        result = self.batch_norm(result)
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


class Multimodal_Soft_GLUMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=3):
        super(Multimodal_Soft_GLUMLP, self).__init__()

        first_layer_embeddins = 20
        # First linear layer
        self.first_layer = nn.Linear(1, first_layer_embeddins)
        self.image_bach_norm = nn.BatchNorm1d(120)

        glu_dim = hidden_dim * first_layer_embeddins

        # GLU layers
        self.glu_layers = nn.ModuleList([just_GELU(glu_dim) for _ in range(num_layers)])

        self.efficient_net = EfficientNet_pretrained_encoder()

        # Last linear layer
        self.last_layer = nn.Linear(hidden_dim * first_layer_embeddins, output_dim)

        self.loss_layer = nn.Linear(hidden_dim * first_layer_embeddins, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, img):
        image_embeddings = self.efficient_net(img)
        image_embeddings = self.image_bach_norm(image_embeddings)

        x = torch.cat([x, image_embeddings], dim=1)
        # Apply the first linear layer

        x = self.first_layer(x.unsqueeze(dim=2)).flatten(start_dim=1)
        x = self.dropout(x)

        # Apply the GLU layers
        for glu in self.glu_layers:
            x = glu(x)

        total_embeddings = x
        # Apply the last linear layer
        logits = self.last_layer(x)
        prob = torch.sigmoid(logits)

        loss_pred = self.loss_layer(x)

        return prob, loss_pred, image_embeddings, total_embeddings

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
