import scipy.interpolate as interpolate
from SALib.sample import sobol_sequence
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import pickle
from torch.optim.lr_scheduler import StepLR
# from functorch import jacrev, vmap, make_functional, grad, vjp
import torch.autograd.functional as F
import timeit
import argparse
import os

from utils import plot_setup, bilinear_interpol, plot_eval
from net import PhysicsInformedNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(77787)
np.random.seed(77787)
torch.set_default_dtype(torch.float64)


def get_args():
    parser = argparse.ArgumentParser(description="original pinn homgeneous modeling")
    parser.add_argument("--name", "-j", type=str, help="experiment name")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="choose a cuda")
    parser.add_argument("--data", "-d", type=str, default="", help="ground truth data path")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dump_folder = f"/home/stan/fullbodyring-pinn/data/{args.name}"
    fig_dir = f"/home/stan/fullbodyring-pinn/data/{args.name}/figs"
    ckpt_dir = f"/home/stan/fullbodyring-pinn/data/{args.name}/ckpt"
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
    
    ## the orignal problem size is 1500x1500 (m) , sos = 600 m/s
    ## scaled down by a factor of 600
    ## equivalent problem size is 2.5x2.5 (m), sos = 1m/s
    kernel_size = 200
    xz_scl = 600
    sos = 600
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

    n_slices = 1000
    time_span = 2.5
    indices = [200, 300, 400, 700, 900]
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
    t_ini1 = 0.0 * np.ones((n_ini**2, 1), dtype=np.float64)
    t_ini2 = (time_pts[1] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    # for enforcing the disp I.C
    X_ini1 = np.concatenate((x_ini, z_ini, t_ini1), axis=1)  # [1600, 3]
    # for enforcing the sec I.C, another snapshot of specfem
    X_ini2 = np.concatenate((x_ini, z_ini, t_ini2), axis=1)  # [1600, 3]
    xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [1600, 2]

    # uploading the wavefields from specfem
    p_scl = max(abs(np.min(indices[0])), abs(np.max(indices[0])))
    p_ini1 = interpolate.griddata(
        xz_0, wave_data[indices[0]].reshape(-1), xz_ini, fill_value=0.0
    )
    p_ini2 = interpolate.griddata(
        xz_0, wave_data[indices[1]].reshape(-1), xz_ini, fill_value=0.0
    )
    p_ini1 = p_ini1.reshape(-1, 1) / p_scl
    p_ini2 = p_ini2.reshape(-1, 1) / p_scl
    p_color = max(abs(np.min(p_ini1)), abs(np.max(p_ini1)))
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
    t_ini1_s2 = 0.0 * np.ones((n_ini**2, 1), dtype=np.float64)
    t_ini2_s2 = (time_pts[1] - time_pts[0]) * np.ones((n_ini**2, 1), dtype=np.float64)
    X_ini1_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini1_s2), axis=1)  # [1600, 3]
    X_ini2_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini2_s2), axis=1)  # [1600, 3]
    xz_ini_s2 = np.concatenate((x_ini_s2, z_ini_s2), axis=1)  # [1600, 2]
    p_ini1_s2 = interpolate.griddata(
        xz_0, wave_data[indices[0]].reshape(-1), xz_ini_s2, fill_value=0.0
    )
    p_ini2_s2 = interpolate.griddata(
        xz_0, wave_data[indices[1]].reshape(-1), xz_ini_s2, fill_value=0.0
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
    xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

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
        "p_color": p_color,
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
    }

     # train
    print("====== Start train Now ... =======")

    model = PhysicsInformedNN(**model_kwargs)
    model.train_adam(n_iters=30001, IfIni=True)
    print("============================================================")

    model.train_LBFGS()
    print("============================================================")

    checkpoints_path = f"{ckpt_dir}/loop_{i}_ini_LBFGS_checkpoints_8000.dump"
    model_kwargs["model_path"] = checkpoints_path
    model = PhysicsInformedNN(**model_kwargs)
    model.train_adam(
        n_iters=30001, calc_NTK=True, update_lambda=True, IfIni=False
    )
