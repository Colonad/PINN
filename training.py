# training.py
import time
import torch
import torch.optim as optim

class PINNTrainer:
    """
    Trainer class that handles:
      - PDE/BC losses
      - Collocation points
      - Optimization (Adam or others)
      - GPU usage & timing
    """
    def __init__(self, model, pde_problem,
                 lambda_bc=1.0, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.pde_problem = pde_problem
        self.lambda_bc = lambda_bc
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.history = {
            "epoch": [],
            "loss_total": [],
            "loss_pde": [],
            "loss_bc": []
        }

    def train(self, x_r, x_left, x_right, epochs=20000, log_interval=1000):
        x_r = x_r.to(self.device)
        x_left = x_left.to(self.device)
        x_right = x_right.to(self.device)

        t0 = time.perf_counter()

        for epoch in range(epochs+1):
            self.optimizer.zero_grad()

            # PDE loss
            u_xx = self.model.second_derivative(x_r)
            f_x = self.pde_problem.source_function(x_r).to(self.device)
            loss_pde = ((-u_xx - f_x)**2).mean()

            # BC loss
            u_l = self.model(x_left)
            u_r = self.model(x_right)
            loss_bc = (u_l**2 + u_r**2).mean()

            # total loss
            loss_total = loss_pde + self.lambda_bc*loss_bc
            loss_total.backward()
            self.optimizer.step()

            # Logging
            if epoch % log_interval == 0:
                self.history["epoch"].append(epoch)
                self.history["loss_total"].append(loss_total.item())
                self.history["loss_pde"].append(loss_pde.item())
                self.history["loss_bc"].append(loss_bc.item())

        t1 = time.perf_counter()
        elapsed = t1 - t0
        return elapsed

    def compute_errors(self, n_eval=200):
        x_eval = torch.linspace(0,1,n_eval).unsqueeze(1).to(self.device)
        x_eval.requires_grad_(True)

        with torch.no_grad():
            u_pred = self.model(x_eval).squeeze()
            u_exact = self.pde_problem.exact_solution(x_eval).squeeze()
            l2_err = torch.sqrt(torch.mean((u_pred - u_exact)**2)).item()

        # approximate derivative
        u_pred_sum = self.model(x_eval).sum()
        du_pred = torch.autograd.grad(
            u_pred_sum, x_eval, create_graph=False
        )[0].squeeze()
        du_exact = (torch.pi*torch.cos(torch.pi*x_eval)).squeeze()
        h1_err = torch.sqrt(torch.mean((du_pred - du_exact)**2)).item()
        x_eval.requires_grad_(False)

        return l2_err, h1_err

    def get_history(self):
        return self.history
