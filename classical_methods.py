# classical_methods.py
import time
import numpy as np
import torch
from numpy.linalg import solve

class FDSolver1D:
    """
    A classical finite difference solver for the 1D Poisson equation:
        -u''(x) = f(x),   x in (0,1),
        u(0)=0, u(1)=0,
    with f(x) = pi^2 sin(pi x).
    """
    def __init__(self, pde_problem, N=50):
        """
        Args:
            pde_problem: an instance of Poisson1DProblem
            N (int): number of intervals in [0,1]
                     -> we have N+1 grid points (including boundaries).
        """
        self.pde = pde_problem
        self.N = N  # number of intervals, => N+1 nodes

        # placeholders
        self.x_grid = None
        self.u_approx = None
        self.build_grid()

    def build_grid(self):
        """
        Build a uniform grid x_0=0, x_N=1, spacing dx=1/N.
        """
        self.x_grid = np.linspace(0.0, 1.0, self.N+1)

    def solve(self):
        """
        Construct the FD system A*u = b, apply Dirichlet BCs, solve for {u_i}.
        """
        start = time.perf_counter()

        dx = 1.0 / self.N
        N_inner = self.N - 1  # interior points i=1..N-1

        # Construct A (N-1)x(N-1) tri-diagonal
        # Each row: -1, +2, -1 on the diagonal => corresponds to central difference
        diag_main  = np.full(N_inner,  2.0)
        diag_upper = np.full(N_inner-1, -1.0)
        diag_lower = np.full(N_inner-1, -1.0)

        # We'll do system -1/dx^2 [u_{i+1} -2u_i + u_{i-1}] = f_i
        # rearranging => (2/dx^2)*u_i - (1/dx^2)*u_{i+1} - ... = f_i
        # but we want the negative Laplacian = f, so effectively we do:
        #  A * u = b,   A = (1/dx^2)* ...
        # We'll incorporate the negative sign properly in the RHS.

        # We'll build a matrix A, then multiply by (1/dx^2):
        A = np.zeros((N_inner, N_inner))
        for i in range(N_inner):
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < N_inner-1:
                A[i, i+1] = -1.0

        # multiply by 1/dx^2 to get final matrix for -u'' = f
        A = (1.0/dx**2) * A

        # Construct b vector
        # b_i = f(x_i), i=1..N-1
        # Because of BCs u(0)=0 and u(N)=0, there's no extra terms from boundaries
        b = np.zeros(N_inner)
        for i in range(1, self.N):
            xi = self.x_grid[i]
            # PDE is -u''(x)=f(x). The FD approximates -u''(x_i) => A[i-1, :]*u.
            # So right side is f(x_i).
            # We'll evaluate f_source using the PDE problem => float
            fx = self.pde.source_function(torch.tensor([xi])).item()
            b[i-1] = fx

        # Solve linear system
        u_inner = solve(A, b)

        # Reconstruct full solution with boundary conditions
        u = np.zeros(self.N+1)
        # u[0]=0, u[N]=0
        for i in range(1, self.N):
            u[i] = u_inner[i-1]

        self.u_approx = u
        end = time.perf_counter()
        self.runtime = end - start

    def compute_errors(self):
        """
        Compare self.u_approx to exact solution in L^2 and H^1 (semi) norms
        using discrete approximations on the FD grid.
        """
        if self.u_approx is None:
            raise RuntimeError("No solution found. Call solve() first.")

        # L2 error: ~ sqrt( sum_{i=0..N} (u_i - u*(x_i))^2 * dx )
        dx = 1.0 / self.N
        exact_vals = self.pde.exact_solution(torch.from_numpy(self.x_grid)).numpy()
        diff_sq = (self.u_approx - exact_vals)**2
        l2_error = np.sqrt(np.sum(diff_sq)*dx)

        # H^1 semi-norm: we approximate derivative with forward difference
        # du_approx_i ~ (u_{i+1} - u_i) / dx
        # du_exact_i  ~ derivative of sin(pi x) => pi cos(pi x)
        # We'll compare i=0..N-1 (the intervals).
        sum_dsq = 0.0
        for i in range(self.N):
            # midpoint or left side? We'll do midpoint for better approximation
            # Let's do left side for simplicity
            up_approx = (self.u_approx[i+1] - self.u_approx[i]) / dx
            xm = 0.5*(self.x_grid[i] + self.x_grid[i+1])  # midpoint
            up_exact = np.pi*np.cos(np.pi*xm)  # derivative
            sum_dsq += (up_approx - up_exact)**2

        h1_error = np.sqrt(sum_dsq*dx)
        return l2_error, h1_error
