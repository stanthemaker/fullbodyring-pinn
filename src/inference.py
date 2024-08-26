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

from net import PhysicsInformedNN
from utils import plot_setup, bilinear_interpol, plot_eval

"""
Sample Usage:
python3 inference.py --folder ./3hz_homo --data ../data/0814_f3_homo.npz --cuda 0 --model ./3hz_homo/ckpt/loop_0_adam_checkpoints_20000.dump
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(77787)
np.random.seed(77787)
torch.set_default_dtype(torch.float64)


def get_args():
    parser = argparse.ArgumentParser(description="original pinn homgeneous modeling")
    parser.add_argument("--folder", "-f", type=str, help="folder to dump")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="choose a cuda")
    parser.add_argument("--data", "-d", type=str, default="", help="data path")
    parser.add_argument(
        "--model", "-m", type=str, default="", help="model path to load"
    )
    parser.add_argument("--map", type=str, default="", help="model path to load")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    inf_dir = args.folder
    model_path = args.model
    map_file = args.map

    if not os.path.exists(inf_dir):
        os.mkdir(inf_dir)

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

    indices = [285, 318, 500, 700, 900, 950]
    wave_data = np.load(args.data)["data"]
    p_scl = max(abs(np.max(wave_data[indices[0]])), abs(np.min(wave_data[indices[0]])))
    print(f"u_scl:{p_scl}")

    n_slices = wave_data.shape[0]
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

    ### PDE residuals
    batch_size = 10000
    n_pde = batch_size * 1
    print("batch_size:", batch_size)
    X_pde_sobol = sobol_sequence.sample(n_pde + 1, 3)[1:, :]
    x_pde = X_pde_sobol[:, 0] * (xmax - xmin) + xmin
    z_pde = X_pde_sobol[:, 1] * (zmax - zmin) + zmin
    t_pde = X_pde_sobol[:, 2] * (t_m - t_st)
    X_pde = np.concatenate(
        (x_pde.reshape(-1, 1), z_pde.reshape(-1, 1), t_pde.reshape(-1, 1)), axis=1
    )

    model_kwargs = {
        "inf_dir": inf_dir,
        "device": device,
        "model_path": model_path,
        "xz_scl": xz_scl,
        "X_evals": X_evals,
        "p_evals": p_evals,
        "x_eval": x_eval,
        "z_eval": z_eval,
    }
    ## inference all wavefield in the whole time and spatial domain
    X_all = []
    T = np.linspace(time_pts[0], time_pts[-1], (indices[-1] - indices[0] + 1))
    for t in T:
        X = np.concatenate((x_eval, z_eval, (t - T[0]) * np.ones_like(x_eval)), axis=1)
        X_all.append(X)
    print(len(X_all))
    model = PhysicsInformedNN(**model_kwargs)
    # outputname = os.path.join(inf_dir, "output.npz")
    # model.inference_field(X_all, savepath=outputname)
    figname = os.path.join(inf_dir, "eval.png")
    model.predict_eval(figname=figname)
