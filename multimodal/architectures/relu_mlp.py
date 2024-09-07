import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super(TabularMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropout_rate = dropout_rate

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

# # Example usage
# input_dim = 100
# hidden_dims = [64, 32, 16]
# output_dim = 1  # For regression or binary classification
# dropout_rate = 0.5
#
# model = TabularMLP(input_dim, hidden_dims, output_dim, dropout_rate)
# print(model)
#
# # Dummy input for testing
# input_tensor = torch.randn(32, input_dim)  # Batch size of 32
# output = model(input_tensor)
# print(output.shape)
