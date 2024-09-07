import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        x = self.linear(x)
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class EfficientNet_pretrained_encoder(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet_pretrained_encoder, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        #self.output_dim = nn.Linear(1280, 120)

    def forward(self, x):
        features = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(features)
        x = x.flatten(start_dim=1)
        #x = self.output_dim(x)

        return x


class Multimodal_GLUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(Multimodal_GLUMLP, self).__init__()

        # First linear layer
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # GLU layers
        self.glu_layers = nn.ModuleList([GLU(hidden_dim) for _ in range(num_layers)])

        self.efficient_net = EfficientNet_pretrained_encoder()

        # Last linear layer
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, img):
        print(x.shape, img.shape)
        image_embeddings = self.efficient_net(img)

        x = torch.cat([x, image_embeddings], dim=1)
        # Apply the first linear layer
        x = self.first_layer(x)
        x = F.relu(x)  # You can use other activations if needed

        # Apply the GLU layers
        for glu in self.glu_layers:
            x = glu(x)

        total_embeddings = x
        # Apply the last linear layer
        x = self.last_layer(x)
        x = torch.sigmoid(x)

        return x, image_embeddings, total_embeddings

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
