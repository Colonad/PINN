# main_classical.py
import torch
import matplotlib.pyplot as plt
import numpy as np

from poisson_1d import Poisson1DProblem
from classical_methods import FDSolver1D

def main():
    print("=== Classical Finite Difference Method for 1D Poisson ===")

    # 1. Instantiate the PDE problem
    pde = Poisson1DProblem()

    # 2. Create the FD solver with N intervals
    N_intervals = 50
    solver = FDSolver1D(pde_problem=pde, N=N_intervals)

    # 3. Solve
    solver.solve()
    fd_time = solver.runtime
    print(f"FD solver completed in {fd_time:.6f} seconds.")

    # 4. Compute errors
    l2_err, h1_err = solver.compute_errors()

    print("-------------------------------------------")
    print("Finite Difference Final Results:")
    print(f"  L^2 Error         = {l2_err:.5e}")
    print(f"  H^1 Semi-norm Err = {h1_err:.5e}")
    print(f"  Runtime           = {fd_time:.4f} s")
    print("-------------------------------------------")

    # 5. Plot solution vs. exact
    x_grid = solver.x_grid
    u_fd   = solver.u_approx
    x_torch = torch.from_numpy(x_grid)
    u_exact = pde.exact_solution(x_torch).numpy()

    plt.figure(figsize=(8,5))
    plt.plot(x_grid, u_exact, 'k--', label="Exact")
    plt.plot(x_grid, u_fd, 'ro-', label="FDM")
    plt.title("Finite Difference vs. Exact Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
