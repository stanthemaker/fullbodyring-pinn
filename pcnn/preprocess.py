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
import os

kernel_size = 200
wavefields_path = "/home/stan/data/pinn/pcnn/wavefields"
dump_dir = "/home/stan/data/pinn/data/wavefields"

# xz_scl = 6 / 25
xz_scl = 600
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

s_spec = 2e-5  # specfem time stepsize
t01 = 4500 * s_spec  # initial disp. input at this time from spec
t02 = (
    5000 * s_spec
)  # sec "initial" disp. input at this time from spec instead of enforcing initial velocity
t03 = 9000 * s_spec  # test data for comparing specfem and trained PINNs
t04 = 13000 * s_spec  # test data for comparing specfem and trained PINNs

t_m = t04  # total time for PDE training.
t_st = t01  # this is when we take the first I.C from specfem

###initial conditions for all events
X_spec = np.loadtxt(wavefields_path + "/wavefield_grid_for_dumps.txt")

X_spec = X_spec / 600  # specfem works with meters unit so we need to convert them to Km
X_spec[:, 0:1] = X_spec[:, 0:1]  # scaling the spatial domain
X_spec[:, 1:2] = X_spec[:, 1:2]  # scaling the spatial domain
xz_spec = np.concatenate((X_spec[:, 0:1], X_spec[:, 1:2]), axis=1)

# uploading the wavefields from specfem
wave_filed_dir_list = sorted(os.listdir(wavefields_path))
U0 = [np.loadtxt(wavefields_path + "/" + f) for f in wave_filed_dir_list]

""" First IC and Second IC """
n_ini = 50
xini_min = xmin
xini_max = xmax
x_ini = np.linspace(xini_min, xini_max, n_ini)
z_ini = np.linspace(xini_min, xini_max, n_ini)
x_ini_mesh, z_ini_mesh = np.meshgrid(x_ini, z_ini)
x_ini = x_ini_mesh.reshape(-1, 1)
z_ini = z_ini_mesh.reshape(-1, 1)
t_ini1 = 0.0 * np.ones((n_ini**2, 1), dtype=np.float64)
t_ini2 = (t02 - t01) * np.ones((n_ini**2, 1), dtype=np.float64)
# for enforcing the disp I.C
X_ini1 = np.concatenate((x_ini, z_ini, t_ini1), axis=1)  # [1600, 3]
# for enforcing the sec I.C, another snapshot of specfem
X_ini2 = np.concatenate((x_ini, z_ini, t_ini2), axis=1)  # [1600, 3]
xz_ini = np.concatenate((x_ini, z_ini), axis=1)  # [1600, 2]

# uploading the wavefields from specfem
u_ini1 = interpolate.griddata(xz_spec, U0[0], xz_ini, fill_value=0.0)  # [1600, 2]
u_scl = max(abs(np.min(u_ini1)), abs(np.max(u_ini1)))
# u_scl = u_scl / 10
u_ini1 = u_ini1.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_ini1.npz"), u_ini1)
u1_min = np.min(u_ini1)
u1_max = np.max(u_ini1)
u_color = max(abs(u1_min), abs(u1_max))
print(
    f"shpae of U_ini1: {u_ini1.shape} === min: [{np.min(u_ini1)}] === max: [{np.max(u_ini1)}]"
)

u_ini2 = interpolate.griddata(xz_spec, U0[1], xz_ini, fill_value=0.0)
u_ini2 = u_ini2.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_ini2.npz"), u_ini2)
print(
    f"shpae of U_ini2: {u_ini2.shape} === min: [{np.min(u_ini2)}] === max: [{np.max(u_ini2)}]"
)

""" First IC and Second IC """
x_ini_s2 = np.linspace(xmin, xmax, n_ini)
z_ini_s2 = np.linspace(zmin, zmax, n_ini)
x_ini_s2_mesh, z_ini_s2_mesh = np.meshgrid(x_ini_s2, z_ini_s2)
x_ini_s2 = x_ini_s2_mesh.reshape(-1, 1)
z_ini_s2 = z_ini_s2_mesh.reshape(-1, 1)
t_ini1_s2 = 0.0 * np.ones((n_ini**2, 1), dtype=np.float64)
t_ini2_s2 = (t02 - t01) * np.ones((n_ini**2, 1), dtype=np.float64)
# for enforcing the disp I.C
X_ini1_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini1_s2), axis=1)  # [1600, 3]
# for enforcing the sec I.C, another snapshot of specfem
X_ini2_s2 = np.concatenate((x_ini_s2, z_ini_s2, t_ini2_s2), axis=1)  # [1600, 3]
xz_ini_s2 = np.concatenate((x_ini_s2, z_ini_s2), axis=1)  # [1600, 2]

# uploading the wavefields from specfem
u_ini1_s2 = interpolate.griddata(xz_spec, U0[0], xz_ini_s2, fill_value=0.0)  # [1600, 2]
u_ini1_s2 = u_ini1_s2.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_ini1_s2.npz"), u_ini1_s2)

u_ini2_s2 = interpolate.griddata(xz_spec, U0[1], xz_ini_s2, fill_value=0.0)
u_ini2_s2 = u_ini2_s2.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_ini2_s2.npz"), u_ini2_s2)

# wavefields for eval
n_eval = 100
x_eval = np.linspace(xmin, xmax, n_eval)
z_eval = np.linspace(zmin, zmax, n_eval)
x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
x_eval = x_eval_mesh.reshape(-1, 1)
z_eval = z_eval_mesh.reshape(-1, 1)
xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

u_eval1_0 = interpolate.griddata(xz_spec, U0[0], xz_eval, fill_value=0.0)  # [1600, 2]
u_eval1 = u_eval1_0.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_eval1.npz"), u_eval1)

u_eval2_0 = interpolate.griddata(xz_spec, U0[1], xz_eval, fill_value=0.0)
u_eval2 = u_eval2_0.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_eval2.npz"), u_eval2)

u_eval3_0 = interpolate.griddata(xz_spec, U0[2], xz_eval, fill_value=0.0)  # Test data
u_eval3 = u_eval3_0.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_eval3.npz"), u_eval3)

u_eval4_0 = interpolate.griddata(xz_spec, U0[3], xz_eval, fill_value=0.0)  # Test data
u_eval4 = u_eval4_0.reshape(-1, 1) / u_scl
np.savez_compressed(os.path.join(dump_dir, "u_eval4.npz"), u_eval4)
