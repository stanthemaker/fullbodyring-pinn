import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os


def shift_half(array):
    half_size = int(array.size / 2)
    array[:half_size] = array[half_size:]
    array[half_size:] = 0
    return array


def plot_all(**kwargs):
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
    smap = np.load(map_file)["data"] / u_scl
    plt.figure(figsize=(8, 6))
    plt.imshow(smap, aspect="auto")
    plt.colorbar(label="Value")
    plt.title("sos map")

    save_path = os.path.join(fig_dir, "sosmap.png")
    plt.savefig(save_path, dpi=300)
