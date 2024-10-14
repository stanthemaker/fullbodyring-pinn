import scipy.interpolate as interpolate
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import argparse
import os
from scipy.io import savemat
import scipy.stats as stats


from utils import plot_setup, bilinear_interpol, plot_eval
from net import Ultra_PINN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(77787)
np.random.seed(77787)
torch.set_default_dtype(torch.float64)


def get_args():
    parser = argparse.ArgumentParser(description="ultra pinn homgeneous modeling")
    parser.add_argument("--name", "-n", type=str, help="experiment name")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="choose a cuda")
    parser.add_argument("--data", "-d", type=str, default="", help="data path")
    parser.add_argument("--model", "-m", type=str, default="", help="model path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dump_folder = f"/home/stan/fullbodyring-pinn/data/{args.name}"
    fig_dir = f"/home/stan/fullbodyring-pinn/data/{args.name}/figs"
    ckpt_dir = f"/home/stan/fullbodyring-pinn/data/{args.name}/ckpt"
    log_file = os.path.join(dump_folder, f"{args.name}.log")
    model_path = args.model
    wave_data = np.load(args.data)["data"]

    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)
        os.mkdir(fig_dir)
        os.mkdir(ckpt_dir)
    else:
        if os.path.exists(log_file):
            os.remove(log_file)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("Your device is: {}".format(device))

    kernel_size = 200
    xz_scl = 600
    sos = 600
    # PINN的x,z范围
    xmin_spec = 0
    xmax_spec = 1500 / xz_scl
    zmin_spec = 0
    zmax_spec = 1500 / xz_scl

    n_abs = 3
    nx = 100
    dx = xmax_spec / nx
    dz = zmax_spec / nx

    xmin = xmin_spec + dx * n_abs
    xmax = xmax_spec - dx * n_abs
    zmin = zmin_spec + dz * n_abs
    zmax = zmax_spec - dz * n_abs

    n_slices = wave_data.shape[0]
    time_span = 2.5
    indices = [0, 100, 200, 300, 400]
    eval_times = []
    for i in range(len(indices)):
        eval_times.append((indices[i] / n_slices) * time_span)
    print("Eval times: ", eval_times)

    tmax = eval_times[-1]  # total time for PDE training.
    tmin = eval_times[0]  # start time
    x_0 = np.linspace(0, 1500, 500) / xz_scl
    z_0 = np.linspace(0, 1500, 500) / xz_scl
    x_0_mesh, z_0_mesh = np.meshgrid(x_0, z_0)
    x_0 = x_0_mesh.reshape(-1, 1)
    z_0 = z_0_mesh.reshape(-1, 1)
    xz_0 = np.concatenate((x_0, z_0), axis=1)
    p_scl = abs(wave_data).max()

    ## eval points
    n_eval = 100
    x_eval = np.linspace(xmin, xmax, n_eval)
    z_eval = np.linspace(zmin, zmax, n_eval)
    x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
    x_eval = x_eval_mesh.reshape(-1, 1)
    z_eval = z_eval_mesh.reshape(-1, 1)
    xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

    p_evals = []
    X_evals = []
    # from orignal gt data (1000, 500, 500) -> (time_pts,100,100)
    for i in range(len(indices)):
        p = interpolate.griddata(
            xz_0, wave_data[indices[i]].reshape(-1), xz_eval, fill_value=0.0
        )
        p = p.reshape(-1, 1) / p_scl
        p_evals.append(p)

    X_evals = []
    for t in eval_times:
        X_eval = np.concatenate(
            (x_eval, z_eval, (t - tmin) * np.ones_like(x_eval)), axis=1
        )
        X_evals.append(X_eval)
       
    ################### plotting   
    shape = (1, len(eval_times))
    plt.figure(figsize=(3 * shape[1], 3 * shape[0]))
    for i in range(len(eval_times)):
        plt.subplot2grid(shape, (0, i))
        plt.scatter(
            x_eval * xz_scl,
            z_eval * xz_scl,
            c=p_evals[i],
            alpha=1,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=10,
            vmin=-1,
            vmax=1,
        )
        plt.axis("equal")
        plt.colorbar()
        plt.title("Specfem t=" + str(eval_times[i]))

    save_path = os.path.join(fig_dir, "pressurefield_eval.png")
    plt.savefig(save_path, dpi=300)
    
    ### Define collocation points
        ## define source signal
    def source_f(x, z, t):
        f_central = 3
        delay = 0.4
        h = (2 * (np.pi * f_central * ((t + tmin) - delay))) * np.exp(
            -((np.pi * f_central * ((t + tmin) - delay)) ** 2)
        )
        _lambda = (sos / xz_scl) / f_central
        sigma = _lambda / 10
        center = xmax_spec / 2
        g = np.exp(
            -((x - center) ** 2 / (2 * sigma**2) + (z - center) ** 2 / (2 * sigma**2))
        )
        return g * h
    
    ### collocation points sampled around center source
    n_center_smaples = 5000
    mu_x, mu_z, mu_t = (
        xmax_spec / 2,
        zmax_spec / 2,
        0.4,
    )
    sigma_x, sigma_z, sigma_t = (
        xmax_spec / 20,
        zmax_spec / 20,
        time_span / 5,
    )
    center_samples = []

    x = stats.truncnorm(
    (xmin - mu_x) / sigma_x, (xmax - mu_x) / sigma_x, loc=mu_x, scale=sigma_x).rvs(size=n_center_smaples)

    z = stats.truncnorm(
    (zmin - mu_z) / sigma_z, (zmax - mu_z) / sigma_z, loc=mu_z, scale=sigma_z).rvs(size=n_center_smaples)

    t = stats.truncnorm(
    (tmin - mu_t) / sigma_t, (tmax - mu_t) / sigma_t, loc=mu_t, scale=sigma_t).rvs(size=n_center_smaples)
    f = source_f(x,z,t) 
    center_samples = np.column_stack((x,z,t,f))

    ### collocation points by sobol sampling
    n_sobol_samples  = 5000
    X_pde_sobol = sobol_sequence.sample(n_sobol_samples + 1, 3)[1:, :]
    x = X_pde_sobol[:, 0] * (xmax - xmin) + xmin
    z = X_pde_sobol[:, 1] * (zmax - zmin) + zmin
    t = X_pde_sobol[:, 2] * (tmax - tmin)
    f = source_f(x,z,t)
    sobol_samples = np.column_stack((x,z,t,f))

    X_pde = np.vstack((center_samples, sobol_samples))
    print("X_pde shape:", X_pde.shape)

    ### colloc points for NTK calculation 
        ### pde NTK
    kernel_size = 200
    X_pde_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
    X_pde_NTK[:, 0] = X_pde_NTK[:, 0] * (xmax - xmin) + xmin
    X_pde_NTK[:, 1] = X_pde_NTK[:, 1] * (zmax - zmin) + zmin
    X_pde_NTK[:, 2] = X_pde_NTK[:, 2] * (tmax - tmin)
        ### ini NTK
    X_ini_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
    X_ini_NTK[:, 0] = X_ini_NTK[:, 0] * (xmax - xmin) + xmin
    X_ini_NTK[:, 1] = X_ini_NTK[:, 1] * (zmax - zmin) + zmin
    X_ini_NTK[:, 2] = X_ini_NTK[:, 2] * (tmax - tmin)
    ### save colloc points and signal for check

    # def f_np(x, z, t):
    #     f_central = 3
    #     delay = 0.4
    #     h = (2 * (np.pi * f_central * ((t + t_st) - delay))) * np.exp(
    #         -((np.pi * f_central * ((t + t_st) - delay)) ** 2)
    #     )
    #     _lambda = (sos / xz_scl) / f_central
    #     sigma = _lambda / 10
    #     center = xmax_spec / 2
    #     g = np.exp(
    #         -((x - center) ** 2 / (2 * sigma**2) + (z - center) ** 2 / (2 * sigma**2))
    #     )
    #     return g * h

    # src_sample = []
    # for sample in samples_3d:
    #     x, z, t = sample
    #     p = f_np(x, z, t)
    #     src_sample.append({"coord": [x, z, t], "val": p})

    # data_dict = {"colloc": X_pde, "src_sample": src_sample}
    # savemat("data.mat", data_dict)
    # print(p_evals[0])
    # print(wave_data[0])
    # exit()
    model_kwargs = {
        "model_path": model_path,
        "log_file": log_file,
        "kernel_size": kernel_size,
        "fig_dir": fig_dir,
        "ckpt_dir": ckpt_dir,
        "device": device,
        "xz_scl": xz_scl,
        "xmax": xmax,
        "xmin": xmin,
        "zmax": zmax,
        "zmin": zmin,
        "X_pde": X_pde,
        "X_pde_NTK":X_pde_NTK,
        "X_ini_NTK":X_ini_NTK,
        "eval_times": eval_times,
        "X_evals": X_evals,
        "p_evals": p_evals,
        "X_ini": X_evals[0], #for now ini is time = 0
        "p_ini": p_evals[0], #for now ini is time = 0
        "source_func": source_f,
    }

    # print("====== Start train Now ... =======")

    model = Ultra_PINN(**model_kwargs)
    model.train_adam(n_iters=40001)

