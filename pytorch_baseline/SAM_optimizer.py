import torch
import torch.nn as nn
import torch.optim as optim


class SAM:
    def __init__(self, model, base_optimizer, rho=0.05, **kwargs):
        self.model = model
        self.base_optimizer = base_optimizer(self.model.parameters(), **kwargs)
        self.rho = rho

    def first_step(self, gradients):
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), gradients):
                if g is None:
                    continue
                scale = self.rho / (torch.norm(g) + 1e-12)
                e_w = scale * g
                p.add_(e_w)  # Perform the ascent step

    def second_step(self, gradients):
        self.base_optimizer.zero_grad()
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), gradients):
                if g is None:
                    continue
                p.sub_(2 * g * self.rho / (torch.norm(g) + 1e-12))  # Restore original weights

    def step(self, closure):
        # Perform the first forward-backward pass
        loss = closure()
        gradients = [p.grad.clone() for p in self.model.parameters() if p.grad is not None]

        self.first_step(gradients)  # First step

        # Perform the second forward-backward pass
        loss = closure()
        loss.backward()

        gradients = [p.grad.clone() for p in self.model.parameters() if p.grad is not None]
        self.second_step(gradients)  # Second step

        self.base_optimizer.step()  # Apply the update to the weights

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# Example usage with a simple model and optimizer
model = nn.Sequential(nn.Linear(10, 1))
optimizer = SAM(model, optim.SGD, lr=0.01, momentum=0.9)


def closure():
    optimizer.zero_grad()
    output = model(torch.randn(10))
    loss = (output - torch.randn(1)).pow(2).sum()
    loss.backward()
    return loss


# Training loop
for epoch in range(100):
    optimizer.step(closure)
