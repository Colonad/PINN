# main.py
import torch
import matplotlib.pyplot as plt

from poisson_1d import Poisson1DProblem
from models import PINN1D
from training import PINNTrainer

def main():
    # 1. Detect device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 2. Instantiate PDE problem
    pde = Poisson1DProblem()

    # 3. Build the PINN model
    model = PINN1D(layers=3, neurons=20, activation='tanh')

    # 4. Prepare domain points
    x_r = pde.domain_collocation_points(n_points=50, seed=42)  # PDE residual pts
    x_left, x_right = pde.boundary_points()                    # boundary coords

    # 5. Create trainer
    trainer = PINNTrainer(
        model=model,
        pde_problem=pde,
        lambda_bc=1.0,    # boundary weight
        lr=1e-3,          # learning rate
        device=device
    )

    # 6. Train the model
    epochs = 20000
    print(f"Training for {epochs} epochs...")
    train_time = trainer.train(x_r, x_left, x_right, epochs=epochs, log_interval=1000)
    print(f"Training completed in {train_time:.2f} seconds.")

    # 7. Compute final PDE / BC / total losses
    history = trainer.get_history()
    final_total_loss = history["loss_total"][-1]
    final_pde_loss   = history["loss_pde"][-1]
    final_bc_loss    = history["loss_bc"][-1]

    # 8. Compute final L^2 and H^1 errors
    l2_error, h1_error = trainer.compute_errors(n_eval=200)

    # 9. Print summary
    print("-------------------------------------------")
    print("Final Results:")
    print(f"  PDE Loss          = {final_pde_loss:.3e}")
    print(f"  BC Loss           = {final_bc_loss:.3e}")
    print(f"  Total Loss        = {final_total_loss:.3e}")
    print(f"  L^2 Error         = {l2_error:.3e}")
    print(f"  H^1 Semi-norm Err = {h1_error:.3e}")
    print(f"  Training Time     = {train_time:.2f} s")
    print("-------------------------------------------")

    # 10. Plot training history
    plot_training_history(history)

    # 11. Plot final PINN solution vs exact
    plot_final_solution(trainer, pde, device)

def plot_training_history(history):
    """
    Plots PDE, BC, and total losses on a log scale vs. training epoch.
    """
    epochs_logged = history["epoch"]
    total_losses  = history["loss_total"]
    pde_losses    = history["loss_pde"]
    bc_losses     = history["loss_bc"]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_logged, total_losses, 'b-',  label="Total Loss")
    plt.plot(epochs_logged, pde_losses,   'r--', label="PDE Loss")
    plt.plot(epochs_logged, bc_losses,    'g-.', label="BC  Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()

def plot_final_solution(trainer, pde, device):
    """
    Evaluate trained model vs. exact solution on [0,1], then plot.
    """
    x_plot = torch.linspace(0,1,200).unsqueeze(1).to(device)
    with torch.no_grad():
        u_pred = trainer.model(x_plot).squeeze()
        u_ex   = pde.exact_solution(x_plot).squeeze()

    x_np    = x_plot.cpu().numpy().flatten()
    u_predn = u_pred.cpu().numpy().flatten()
    u_exn   = u_ex.cpu().numpy().flatten()

    plt.subplot(1,2,2)
    plt.plot(x_np, u_exn,   'k--', label="Exact")
    plt.plot(x_np, u_predn, 'r-',  label="PINN")
    plt.title("PINN Approx vs. Exact")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
