import torch
import torch.nn as nn
from rphpgd import RPHPGD
from torch import tensordot
from scipy.stats import ortho_group
from torch.linalg import solve, norm, matrix_norm, vector_norm
from torch.optim import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from numpy.random import choice
from pgd import PGD
from math import log
import os
import multiprocessing as mp

# Compute the 4-order tensor product
def tensorproduct(bases):
    Tensor = None
    for base in bases:
        if (Tensor is None):
            Tensor = torch.tensordot(base, torch.tensordot(base, torch.tensordot(base, base, dims=0), dims=0), dims=0)
        else:
            Tensor += torch.tensordot(base, torch.tensordot(base, torch.tensordot(base, base, dims=0), dims=0), dims=0)
    return Tensor

@torch.no_grad()
def normalize(bases):
    for i in range(len(bases)):
        bases[i].mul_(1/vector_norm(bases[i]))

def objective(predicted_tensor, target_tensor):
    return norm(target_tensor - predicted_tensor)**2

# closure function for computing approximate hessian-vector product
def closure(delta, d, point, target_tensor, objective):
    _point = point.detach().clone()
    _point.view(-1).add_(d, alpha=delta)
    _point.requires_grad = True

    _predicted_tensor = tensorproduct(_point)
    _reconstruction_error = objective(_predicted_tensor, target_tensor)
    _reconstruction_error.backward()

    return torch.flatten(_point.grad.detach().clone())

def run(optimizer_type, use_scheduler = False, init_point = None, iterations = 60000, target_tensor = None):
    point = init_point.clone()
    point.requires_grad = True

    # optimizer
    tolerance = 9
    PGD_perturb_interval = 30
    radius = 0.01
   
    if (optimizer_type == "SGD"):
        optimizer = SGD([point], lr=lr)
    elif (optimizer_type == "PGD"):
        optimizer = PGD([point], lr=lr, tolerance=tolerance, perturb_interval=PGD_perturb_interval, radius=radius)
    elif (optimizer_type == "RPHPGD"):
        optimizer = RPHPGD([point], lr = lr, 
                                tolerance=tolerance, 
                                PGD_interval=PGD_perturb_interval, 
                                RPHPGD_interval=2*PGD_perturb_interval, 
                                subspace_dimension=10, 
                                init_perturbation_radius=radius,
                                sigma=1e-2, 
                                delta=1, 
                                buffer_size=5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 250, gamma = 0.95)

    reconstruction_errors = []
    gradient_size = []
    target_tensor_norm = norm(target_tensor)**2
    PGD_trigger = []
    RPHPGD_trigger = []
    grad_norms = []

    for i in trange(iterations):
        optimizer.zero_grad()

        predicted_tensor = tensorproduct(point)

        reconstruction_error = objective(predicted_tensor, target_tensor)

        reconstruction_error.backward()

        if (optimizer_type == "RPHPGD"):
            PGD_flag, RPHPGD_flag, grad_norm = optimizer.step(closure = closure, point = point, target_tensor = target_tensor, objective = objective)
            # For plot
            if (PGD_flag):
                PGD_trigger.append(i)
            if (RPHPGD_flag):
                RPHPGD_trigger.append(i)
        else:
            optimizer.step()

        normalize(point)

        grad_norms.append(grad_norm.item())
        reconstruction_errors.append(log((reconstruction_error / target_tensor_norm).to("cpu").detach().numpy(),10))

        if (use_scheduler):
            scheduler.step()

    return reconstruction_errors, PGD_trigger, RPHPGD_trigger, grad_norms

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dtype = torch.double
lr = 1e-2
if __name__ == "__main__":
    epoches = 1
    tensor_size = 10
    iterations = 25000
    for epoch in range(epoches):
        init_point = torch.randn((tensor_size, tensor_size), device=device, dtype=dtype)
        for i in range(tensor_size):
            init_point[i] = init_point[i] / vector_norm(init_point[i])
        #########################################
        # Orthonormal vectors and target tensor
        bases = ortho_group.rvs(tensor_size)
        bases = torch.from_numpy(bases).to(device)
        target_tensor = tensorproduct(bases)
        #########################################
        tasks = [("SGD", False, init_point, iterations, target_tensor),
                 ("SGD", True, init_point, iterations, target_tensor),
                 ("PGD", False, init_point, iterations, target_tensor),
                 ("PGD", True, init_point, iterations, target_tensor),
                 ("RPHPGD", False, init_point, iterations, target_tensor),
                 ("RPHPGD", True, init_point, iterations, target_tensor)]
        
        reconstruction_errors, PGD_trigger, RPHPGD_trigger, grad_norms = run(*tasks[5])
        # with mp.Pool(processes = 2) as pool:
        #     result = pool.starmap(run, [tasks[4]])

        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title("reconstruction error")
        ax2.set_title("gradient size")
        ax1.plot(reconstruction_errors)
        ax1.vlines(PGD_trigger, ymin=-1, ymax=1, colors='C0', linestyles="dotted", label="PGD trigger")
        ax1.vlines(RPHPGD_trigger, ymin=-1, ymax=1, colors='C1', linestyles="dotted", label="RPHPGD trigger")
        ax2.plot(grad_norms)
        ax1.legend()
        # plot2.legend()
        # filename = f"result/result_iijj_{epoch}.png"
        # plt.savefig(filename,format="png")
        plt.show()