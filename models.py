# models.py
import torch
import torch.nn as nn

class PINN1D(nn.Module):
    """
    A feedforward neural network (PINN) for 1D Poisson problems.
    """
    def __init__(self, layers=3, neurons=20, activation='tanh'):
        super().__init__()
        self.layers = layers
        self.neurons = neurons
        self.activation = activation.lower()

        # Choose activation
        if self.activation == 'tanh':
            self.act_fn = torch.tanh
        elif self.activation == 'relu':
            self.act_fn = nn.ReLU()
        else:
            raise ValueError("Supported activations: 'tanh', 'relu'")

        # Build network layers
        self.linears = nn.ModuleList()
        input_dim = 1
        output_dim = 1

        # First layer
        self.linears.append(nn.Linear(input_dim, neurons))
        # Hidden layers
        for _ in range(layers-1):
            self.linears.append(nn.Linear(neurons, neurons))
        # Output layer
        self.linears.append(nn.Linear(neurons, output_dim))

        # Initialize weights (Xavier)
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = x
        for i in range(len(self.linears)-1):
            out = self.linears[i](out)
            out = self.act_fn(out)
        out = self.linears[-1](out)
        return out

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        u = self.forward(x)
        grad_u = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        grad2_u = torch.autograd.grad(
            grad_u, x,
            grad_outputs=torch.ones_like(grad_u),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad2_u
