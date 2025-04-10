# poisson_1d.py
import torch
import numpy as np

class Poisson1DProblem:
    """
    Encapsulates the 1D Poisson PDE problem:
        -u''(x) = pi^2 sin(pi x) for x in (0,1),
        with boundary conditions u(0) = 0, u(1) = 0.
    """
    def __init__(self):
        pass

    def source_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side: f(x) = pi^2 sin(pi x).
        """
        return (np.pi**2) * torch.sin(np.pi * x)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact solution: u*(x) = sin(pi x).
        """
        return torch.sin(np.pi * x)

    def domain_collocation_points(self, n_points=50, seed=0) -> torch.Tensor:
        """
        Generate interior collocation points in (0,1).
        Excludes boundaries for PDE residual enforcement.
        """
        torch.manual_seed(seed)
        # uniform sampling for simplicity
        x_r = torch.linspace(0., 1., n_points+2)[1:-1]
        return x_r

    def boundary_points(self):
        """
        Returns boundary coordinates for x=0 and x=1.
        """
        return torch.tensor([0.0]), torch.tensor([1.0])
