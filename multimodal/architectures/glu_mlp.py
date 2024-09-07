import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GLUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GLUMLP, self).__init__()

        # First linear layer
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # GLU layers
        self.glu_layers = nn.ModuleList([GLU(hidden_dim) for _ in range(num_layers)])

        # Last linear layer
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Apply the first linear layer
        x = self.first_layer(x)
        x = F.relu(x)  # You can use other activations if needed

        # Apply the GLU layers
        for glu in self.glu_layers:
            x = glu(x)

        # Apply the last linear layer
        x = self.last_layer(x)
        x = torch.sigmoid(x)
        return x

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
