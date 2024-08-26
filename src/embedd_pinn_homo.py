import scipy.interpolate as interpolate
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.optim.lr_scheduler import StepLR
from functorch import jacrev, vmap, make_functional, grad, vjp
import torch.autograd.functional as F
import timeit
import argparse
import os

from utils import plot_setup, bilinear_interpol, plot_eval
from net import Embed_PINN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(77787)
np.random.seed(77787)
torch.set_default_dtype(torch.float64)


def get_args():
    parser = argparse.ArgumentParser(description="embedded pinn homgeneous modeling")
    parser.add_argument("--name", "-j", type=str, help="experiment name")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="choose a cuda")
    parser.add_argument("--data", "-d", type=str, default="", help="data path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dump_folder = f"/home/stan/data/pinn/pcnn/{args.name}"
    fig_dir = f"/home/stan/data/pinn/pcnn/{args.name}/figs"
    ckpt_dir = f"/home/stan/data/pinn/pcnn/{args.name}/ckpt"
    wavefields_path = "/home/stan/data/pinn/pcnn/wavefields"
    log_file = os.path.join(dump_folder, f"{args.name}.log")
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

    n_slices = 600
    indices = [0, 100, 200, 300]
    time_span = 2.5
    time_pts = []
    for i in range(len(indices)):
        time_pts.append((indices[i] / n_slices) * time_span)

    t_m = time_pts[-1]  # total time for PDE training.
    t_st = time_pts[0]  # this is when we take the first I.C from specfem
    x_0 = np.linspace(0, 1500, 500) / xz_scl
    z_0 = np.linspace(0, 1500, 500) / xz_scl
    x_0_mesh, z_0_mesh = np.meshgrid(x_0, z_0)
    x_0 = x_0_mesh.reshape(-1, 1)
    z_0 = z_0_mesh.reshape(-1, 1)
    xz_0 = np.concatenate((x_0, z_0), axis=1)

    """ First IC and Second IC """
    n_ini = 80
    xini_min = xmin
    xini_max = xmax
    x_ini = np.linspace(xini_min, xini_max, n_ini)
    z_ini = np.linspace(xini_min, xini_max, n_ini)
    x_ini_mesh, z_ini_mesh = np.meshgrid(x_ini, z_ini)
    x_ini = x_ini_mesh.reshape(-1, 1)
    z_ini = z_ini_mesh.reshape(-1, 1)
    t_ini1 = (time_pts[-2] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    t_ini2 = (time_pts[-1] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    # for enforcing the disp I.C
    X_ini1 = np.concatenate((x_ini, z_ini, t_ini1), axis=1)  # [n_ini ** 2, 3]
    # for enforcing the sec I.C, another snapshot of specfem
    X_ini2 = np.concatenate((x_ini, z_ini, t_ini2), axis=1)  # [n_ini ** 2, 3]
    xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [n_ini ** 2, 2]

    p_scl = abs(wave_data).max()
    u_color = 1.0
    p_ini1 = interpolate.griddata(
        xz_0, wave_data[indices[-2]].reshape(-1), xz_ini, fill_value=0.0
    )
    p_ini2 = interpolate.griddata(
        xz_0, wave_data[indices[-1]].reshape(-1), xz_ini, fill_value=0.0
    )
    p_ini1 = p_ini1.reshape(-1, 1) / p_scl
    p_ini2 = p_ini2.reshape(-1, 1) / p_scl
    print(f"p_scl:{p_scl}")
    print(
        f"shpae of P_ini1: {p_ini1.shape} === min: [{np.min(p_ini1)}] === max: [{np.max(p_ini1)}]"
    )
    print(
        f"shpae of P_ini2: {p_ini2.shape} === min: [{np.min(p_ini2)}] === max: [{np.max(p_ini2)}]"
    )
    """ First IC and Second IC """
    x_ini_s2 = np.linspace(xmin, xmax, n_ini)
    z_ini_s2 = np.linspace(zmin, zmax, n_ini)
    x_ini_s2_mesh, z_ini_s2_mesh = np.meshgrid(x_ini_s2, z_ini_s2)
    x_ini_s2 = x_ini_s2_mesh.reshape(-1, 1)
    z_ini_s2 = z_ini_s2_mesh.reshape(-1, 1)
    t_ini1_s2 = (time_pts[-2] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    t_ini2_s2 = (time_pts[-1] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    X_ini1_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini1_s2), axis=1)  # [1600, 3]
    X_ini2_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini2_s2), axis=1)  # [1600, 3]
    xz_ini_s2 = np.concatenate((x_ini_s2, z_ini_s2), axis=1)  # [1600, 2]
    p_ini1_s2 = interpolate.griddata(
        xz_0, wave_data[indices[-2]].reshape(-1), xz_ini_s2, fill_value=0.0
    )
    p_ini2_s2 = interpolate.griddata(
        xz_0, wave_data[indices[-1]].reshape(-1), xz_ini_s2, fill_value=0.0
    )
    p_ini1_s2 = p_ini1_s2.reshape(-1, 1) / p_scl
    p_ini2_s2 = p_ini2_s2.reshape(-1, 1) / p_scl
    # wavefields for eval
    n_eval = 100
    x_eval = np.linspace(xmin, xmax, n_eval)
    z_eval = np.linspace(zmin, zmax, n_eval)
    x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
    x_eval = x_eval_mesh.reshape(-1, 1)
    z_eval = z_eval_mesh.reshape(-1, 1)
    xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [n_eval**2, 2]

    p_evals = []
    for i in range(len(indices)):
        p = interpolate.griddata(
            xz_0, wave_data[indices[i]].reshape(-1), xz_eval, fill_value=0.0
        )
        p = p.reshape(-1, 1) / p_scl
        p_evals.append(p)

    X_evals = []
    for t in time_pts:
        X_eval = np.concatenate(
            (x_eval, z_eval, (t - time_pts[0]) * np.ones_like(x_eval)), axis=1
        )
        X_evals.append(X_eval)

    ################### plotting
    kwargs = {
        "ini": {
            "x_ini": x_ini,
            "z_ini": z_ini,
            "p_ini1": p_ini1,
            "p_ini2": p_ini2,
        },
        "eval": {
            "x_eval": x_eval,
            "z_eval": z_eval,
            "p_evals": p_evals,
        },
        "xz_scl": xz_scl,
        "time_pts": time_pts,
        "p_color": 1.0,
        "p_scl": p_scl,
        "fig_dir": fig_dir,
    }
    plot_setup(**kwargs)
    ### PDE residuals
    batch_size = 10000
    n_pde = batch_size * 1
    print("kernel_size", ":", batch_size)
    X_pde_sobol = sobol_sequence.sample(n_pde + 1, 3)[1:, :]
    x_pde = X_pde_sobol[:, 0] * (xmax - xmin) + xmin
    z_pde = X_pde_sobol[:, 1] * (zmax - zmin) + zmin
    t_pde = X_pde_sobol[:, 2] * (t_m - t_st)
    X_pde = np.concatenate(
        (x_pde.reshape(-1, 1), z_pde.reshape(-1, 1), t_pde.reshape(-1, 1)), axis=1
    )

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
    n_samples = 5000
    samples_3d = []

    ### add denser sample points
    while len(samples_3d) < n_samples:
        x = np.random.normal(loc=mu_x, scale=sigma_x)
        z = np.random.normal(loc=mu_z, scale=sigma_z)
        t = np.random.normal(loc=mu_t, scale=sigma_t)

        if xmin <= x <= xmax and xmin <= z <= zmax and t_st <= t <= time_span:
            samples_3d.append([x, z, t])

    samples_3d = np.array(samples_3d)
    X_pde = np.concatenate((X_pde, samples_3d), axis=0)
    print("X_pde shape:", X_pde.shape)

    def f(x, z, t):
        f_central = 3
        delay = 0.4
        h = (2 * (torch.pi * f_central * ((t + t_st) - delay))) * torch.exp(
            -((torch.pi * f_central * ((t + t_st) - delay)) ** 2)
        )
        _lambda = (sos / xz_scl) / f_central
        sigma = _lambda / 10
        center = xmax_spec / 2
        g = torch.exp(
            -((x - center) ** 2 / (2 * sigma**2) + (z - center) ** 2 / (2 * sigma**2))
        )
        return g * h

    model_kwargs = {
        "log_file": log_file,
        "kernel_size": kernel_size,
        "fig_dir": fig_dir,
        "ckpt_dir": ckpt_dir,
        "device": device,
        "xz_scl": xz_scl,
        "time_pts": time_pts,
        "xmax": xmax,
        "xmin": xmin,
        "zmax": zmax,
        "zmin": zmin,
        "xini_min": xini_min,
        "xini_max": xini_max,
        "ini_cond": {
            "X_ini1": X_ini1,
            "X_ini2": X_ini2,
            "p_ini1": p_ini1,
            "p_ini2": p_ini2,
            "X_ini1_s2": X_ini1_s2,
            "X_ini2_s2": X_ini2_s2,
            "p_ini1_s2": p_ini1_s2,
            "p_ini2_s2": p_ini2_s2,
        },
        "X_pde": X_pde,
        "X_evals": X_evals,
        "p_evals": p_evals,
        "x_eval": x_eval,
        "z_eval": z_eval,
        "f": f,
    }

    print("====== Start train Now ... =======")

    model = Embed_PINN(**model_kwargs)
    model.train_adam(
        n_iters=30001, calc_NTK=True, update_lambda=True, IfIni=False, loop_iter=i
    )
    # K_ini1_list = model.K_ini1_log
    # K_pde_list = model.K_pde_log
    # lambda_ini1_log = model.lambda_ini1_log
    # lambda_pde_log = model.lambda_pde_log
    # print("============================================================")
    # K_ini1_all.append(K_ini1_list)
    # K_pde_all.append(K_pde_list)
    # lambda_ini1_all.append(lambda_ini1_log)
    # lambda_pde_all.append(lambda_pde_log)
