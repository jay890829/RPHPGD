import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from sgd import SGD
from torch.optim.lr_scheduler import StepLR
from pgd import PGD
from rphpgd import RPHPGD
import numpy as np
from tqdm import trange
import torch.multiprocessing as mp
from math import log10
torch.set_printoptions(profile="full")

def objective(x):
        # h = torch.ones(dimension, dtype=x.dtype, device = x.device)
        # h[0] = -1
        # H = torch.diag(h)
        # return 0.5 * torch.matmul(torch.matmul(x.t(), H), x) + 1/16 * x[0]**4 #- 1/2 * x[1]**4 + 1/8 * x[2]**2
        return x[0]**4/16-x[0]**2/2+9/8*x[1]**2
        
def closure(delta, d, x, objective):
    _x = x.detach().clone()
    _x.add_(d, alpha = delta)
    _x.requires_grad = True

    function_value = objective(_x)
    function_value.backward()

    return _x.grad.detach().clone()

grad_norms = []
# This function return function values
def run(objective, init_point, optimizer_type, device, iterations=50000):
    point = torch.tensor(init_point, dtype=torch.double, requires_grad=True, device = device)

    tolerance = 1e-2
    PGD_perturb_interval = 50
    radius = 0.001
   
    if (optimizer_type == SGD):
        optimizer = optimizer_type([point], lr=lr)
    elif (optimizer_type == PGD):
        optimizer = optimizer_type([point], lr=lr, tolerance=tolerance, perturb_interval=PGD_perturb_interval, radius=radius)
    elif (optimizer_type == RPHPGD):
        optimizer = optimizer_type([point], lr = lr, 
                                tolerance=tolerance, 
                                PGD_interval=PGD_perturb_interval, 
                                RPHPGD_interval=2*PGD_perturb_interval, 
                                subspace_dimension=10, 
                                init_perturbation_radius=radius,
                                sigma=1e-2, 
                                delta=1e-5, 
                                buffer_size=3)
    
    function_values = []
    scheduler = StepLR(optimizer=optimizer, step_size=1000, gamma=gamma)
    PGD_perturbated_iter = []
    RPHPGD_perturbated_iter = []
    e_maxs = []
    e_mins = []

    for i in trange(iterations):
        optimizer.zero_grad()
        function_value = objective(point)

        function_value.backward()
        function_values.append(function_value.item())
        if (optimizer_type == RPHPGD):
            PGD_flag, RPHPGD_flag, grad_norm = optimizer.step(closure = closure, x = point, objective = objective)
            # grad_norms.append(grad_norm)
            if (PGD_flag):
                PGD_perturbated_iter.append(i)
            if (RPHPGD_flag):
                RPHPGD_perturbated_iter.append(i)
        else:
            optimizer.step()
        
    if (optimizer_type == SGD):
        return (function_values, "SGD")
    elif (optimizer_type == PGD):
        return (function_values, "PGD")
    elif (optimizer_type == RPHPGD):
        return (function_values, "RPHPGD", PGD_perturbated_iter, RPHPGD_perturbated_iter)
    
# This function return function track
def run_2d(objective, init_point, optimizer_type, device, iterations=50000):
    point = torch.tensor(init_point, dtype=torch.double, requires_grad=True, device = device)

    tolerance = 1e-2
    PGD_perturb_interval = 30
    radius = 1
    if (optimizer_type == SGD):
        optimizer = optimizer_type([point], lr=lr)
    elif (optimizer_type == PGD):
        optimizer = optimizer_type([point], lr=lr, tolerance=tolerance, perturb_interval=PGD_perturb_interval/2, radius=radius)
    elif (optimizer_type == RPHPGD):
        optimizer = optimizer_type([point], lr = lr, 
                                tolerance=tolerance, 
                                PGD_interval=PGD_perturb_interval/2, 
                                RPHPGD_interval=PGD_perturb_interval, 
                                subspace_dimension=2, 
                                init_perturbation_radius=radius,
                                sigma=1e-2, 
                                delta=1e-5, 
                                buffer_size=3)
    
    scheduler = StepLR(optimizer=optimizer, step_size=1000, gamma=gamma)
    track_X = []
    track_Y = []

    for i in trange(iterations):
        optimizer.zero_grad()
        function_value = objective(point)

        function_value.backward()
        if (optimizer_type == RPHPGD):
            PGD_flag, RPHPGD_flag, grad_norm = optimizer.step(closure = closure, x = point, objective = objective)
        else:
            optimizer.step()
        # scheduler.step()
        # Only need the result of 75th iteration
        if (i==75 and optimizer_type == PGD):
            return (point.to("cpu").detach().clone(), "PGD")
        elif (i==75 and optimizer_type == RPHPGD):
            return (point.to("cpu").detach().clone(), "RPHPGD")

# Plotting the function track
def contour_track(track_X, track_Y,i):
    sample_num = 5000
    X, Y = np.meshgrid(np.linspace(start=-3, stop=3, num=sample_num, dtype=np.double), np.linspace(start=-3, stop=3, num=sample_num, dtype=np.double))
    Z = X**4/16 - X**2/2 + 9/8 * Y**2
    levels = [-0.5,0,1,2,3]

    fig, (ax1) = plt.subplots(1, 1)
    CS = ax1.contour(X, Y, Z, levels=levels)
    labels = ax1.clabel(CS, CS.levels)
    ax1.scatter(track_X["PGD"], track_Y["PGD"], label = "PGD", s=1, c="C0")
    ax1.set_title("PGD sampled at t=75")
    plt.savefig(f"./track_90_PGD_{i}.png")
    # plt.show()

    fig, (ax2) = plt.subplots(1, 1)
    CS = ax2.contour(X, Y, Z, levels=levels)
    labels = ax2.clabel(CS, CS.levels)
    ax2.scatter(track_X["RPHPGD"], track_Y["RPHPGD"], label = "RPHPGD", s=1, c="C1")
    ax2.set_title("RPHPGD sampled at t=75")
    plt.savefig(f"./track_90_RPHPGD_{i}.png")
    # plt.show()

# Plotting the function value
def function_value_plot(results):
    xticks = [x for x in range(iterations) if x%1000==0]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for result in results:
        ax1.plot(result[0], label = result[1])
        if (result[1] == "RPHPGD"):
            ax1.vlines(result[2], ymin=-1, ymax=1, colors='C0', linestyles="dotted", label="PGD trigger")
            ax1.vlines(result[3], ymin=-1, ymax=1, colors='C1', linestyles="dotted", label="RPHPGD trigger")
    ax1.legend()
    # ax2.plot(grad_norms)
    # ax2.vlines(result[2], ymin=-1, ymax=1, colors='C0', linestyles="dotted")
    plt.xticks(ticks = xticks, labels = xticks)
    plt.show()
    # plt.savefig(f"./result_lr={lr}_{epoch}.png")

dimension = 2
lr = 0.045
gamma = 0.5

if __name__ == "__main__":
    epoches = 2000
    iterations = 10000
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    # with mp.Pool(processes = 2) as pool:
    #     result = pool.starmap(run, [tasks[2]]*2)
    
    
    task_num = 2
    
    for i in range(10):
        init_point = [np.random.choice((1e-10, -1e-10)), np.random.choice((1e-10, -1e-10))]
        tasks = [(objective, init_point, SGD, device, iterations), 
                (objective, init_point, PGD, device, iterations), 
                (objective, init_point, RPHPGD, device, iterations)]
        results = []
        track_X = {"PGD":[], "RPHPGD":[]}
        track_Y = {"PGD":[], "RPHPGD":[]}
        for epoch in trange(epoches):
            # with mp.Pool(processes = task_num) as pool:
            #     results = pool.starmap(run_2d, tasks[1:3])
            results.append(run_2d(*tasks[1]))
            results.append(run_2d(*tasks[2]))
            for result in results:
                track_X[result[1]].append(result[0][0].item())
                track_Y[result[1]].append(result[0][1].item())
        # function_value_plot(results)
        contour_track(track_X, track_Y,i)

    #===========================Animation===============================#
    # # levels = np.linspace(np.min(Z), np.max(Z), num=10)
    
    # # ax.plot(-2,0,"rX")
    # # ax.plot(2,0,"rX")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_wireframe(X,Y,Z)

    # def update_frame(frame, lines, result):
    #     for i in range(len(lines)):
    #         lines[i].set_data_3d(result[i][0][:frame], result[i][1][:frame], result[i][2][:frame])
    #     return lines
    
    # lines = [ax.plot([],[],[], color="red")[0]]#, ax.plot([],[],[], color="green")[0]]#, ax.plot([],[],[], color="yellow")[0]]
    # ani = animation.FuncAnimation(fig, func = update_frame, frames = iterations, fargs=(lines, result), interval = 10, blit=True)
    # plt.show()