import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os
import torch


def shift_half(array):
    half_size = int(array.size / 2)
    array[:half_size] = array[half_size:]
    array[half_size:] = 0
    return array


def plot_setup(**kwargs):
    # Extract parameters from kwargs
    x_ini = kwargs.get("x_ini")
    z_ini = kwargs.get("z_ini")
    u_ini1 = kwargs.get("u_ini1")
    u_ini2 = kwargs.get("u_ini2")

    x_eval = kwargs.get("x_eval")
    z_eval = kwargs.get("z_eval")
    p_evals = kwargs.get("p_evals")

    xz_scl = kwargs.get("xz_scl")
    time_pts = kwargs.get("time_pts")

    u_color = kwargs.get("u_color")
    map_file = kwargs.get("map_file")
    u_scl = kwargs.get("u_scl")
    fig_dir = kwargs.get("fig_dir")

    # Plot of inputs for the sum of the events (initial conditions)
    ini_time = [0, round(time_pts[1] - time_pts[0], 4)]
    n_eval_time = len(ini_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(3 * shape[1], 3 * shape[0]))
    U_ini_plot = [u_ini1, u_ini2]

    for it in range(n_eval_time):
        plt.subplot2grid(shape, (0, it))
        plt.scatter(
            x_ini * xz_scl,
            z_ini * xz_scl,
            c=U_ini_plot[it],
            alpha=1,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=10,
            vmin=-1,
            vmax=1,
        )
        plt.colorbar()
        plt.title("ini x t=" + str(ini_time[it]))

    save_path = os.path.join(fig_dir, "wavefield_init.png")
    plt.savefig(save_path, dpi=300)

    # Plot of inputs for the sum of the events (evaluated conditions)
    eval_time = [0] + [
        round(time_pts[i] - time_pts[0], 4) for i in range(1, len(time_pts))
    ]
    n_eval_time = len(eval_time)
    shape = (1, n_eval_time)

    plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

    for it in range(len(eval_time)):
        plt.subplot2grid(shape, (0, it))
        plt.scatter(
            x_eval * xz_scl,
            z_eval * xz_scl,
            c=p_evals[it],
            alpha=1,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=10,
            vmin=-u_color,
            vmax=u_color,
        )
        plt.axis("equal")
        plt.colorbar()
        plt.title("Specfem t=" + str(eval_time[it]))

    save_path = os.path.join(fig_dir, "wavefield_eval.png")
    plt.savefig(save_path, dpi=300)

    # Plot sos map
    if not map_file == "":
        smap = np.load(map_file)["data"] / u_scl
        plt.figure(figsize=(8, 6))
        plt.imshow(smap.T, aspect="auto")
        plt.colorbar(label="Value")
        plt.title("sos map")

        save_path = os.path.join(fig_dir, "sosmap.png")
        plt.savefig(save_path, dpi=300)


def plot_eval(**kwargs):
    X_evals = kwargs.get("X_evals")
    x_eval = kwargs.get("x_eval")
    z_eval = kwargs.get("z_eval")
    xz_scl = kwargs.get("xz_scl")
    p_evals = kwargs.get("p_evals")
    P_PINN_pred = kwargs.get("P_PINN_pred")
    P_diff_diff = kwargs.get("P_diff_diff")
    savepath = kwargs.get("savepath")

    n_eval_time = len(X_evals)  # Number of evaluations
    shape = (3, n_eval_time)  # Shape of the plot grid (3 rows)
    fig1 = plt.figure(figsize=(3 * shape[1], 3 * shape[0]))

    s = 10  # Marker size for scatter plot
    for it in range(n_eval_time):
        # Plot true evaluations
        plt.subplot2grid(shape, (0, it))
        plt.scatter(
            x_eval * xz_scl,
            z_eval * xz_scl,
            c=p_evals[it],
            alpha=0.9,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=s,
            vmin=-1,
            vmax=1,
        )
        plt.xticks([])
        plt.yticks([])
        plt.axis("equal")
        plt.colorbar()

        # Plot predictions
        plt.subplot2grid(shape, (1, it))
        plt.scatter(
            x_eval * xz_scl,
            z_eval * xz_scl,
            c=P_PINN_pred[it],
            alpha=0.9,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=s,
            vmin=-1,
            vmax=1,
        )
        plt.xticks([])
        plt.yticks([])
        plt.axis("equal")
        plt.colorbar()

        # Plot differences
        plt.subplot2grid(shape, (2, it))
        plt.scatter(
            x_eval * xz_scl,
            z_eval * xz_scl,
            c=P_diff_diff[it],
            alpha=0.9,
            edgecolors="none",
            cmap="seismic",
            marker="o",
            s=s,
        )
        plt.xticks([])
        plt.yticks([])
        plt.axis("equal")
        plt.colorbar()

    # Save the figure
    plt.savefig(savepath, dpi=100)
    plt.close()


def bilinear_interpol(x, z, data):
    x0 = np.floor(x).astype(int)
    z0 = np.floor(z).astype(int)
    x1 = x0 + 1
    z1 = z0 + 1
    x_frac = x - x0
    z_frac = z - z0

    Q11 = data[x0, z0]
    Q21 = data[x1, z0]
    Q12 = data[x0, z1]
    Q22 = data[x1, z1]

    sos = (
        Q11 * (1 - x_frac) * (1 - z_frac)
        + Q21 * x_frac * (1 - z_frac)
        + Q12 * (1 - x_frac) * z_frac
        + Q22 * x_frac * z_frac
    )
    return sos