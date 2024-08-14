import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def MSEloss(a, b):
    error = np.mean((a - b) ** 2)
    return error


# Load the data
data_apporx = np.load("/home/stan/data/pinn/data/0808_f2.5_4slice.npz")["data"]
data_origin = np.load("/home/stan/data/pinn/data/origin4slice.npz")["data"]

# Create a 4 row * 3 column figure
fig, axs = plt.subplots(4, 3, figsize=(18, 24))

for i in range(4):
    u0_origin = data_origin[i]
    u_scl_origin = max(abs(np.min(u0_origin)), abs(np.max(u0_origin)))
    u0_origin = u0_origin / u_scl_origin

    u0_approx = data_apporx[i]
    u0_approx = zoom(
        u0_approx,
        (
            u0_origin.shape[0] / u0_approx.shape[0],
            u0_origin.shape[1] / u0_approx.shape[1],
        ),
        order=1,
    )
    u_scl_approx = max(abs(np.min(u0_approx)), abs(np.max(u0_approx)))
    u0_approx = u0_approx / u_scl_approx

    u0_diff = u0_origin - u0_approx

    # Plot u0_origin
    cax0 = axs[i, 0].imshow(u0_origin, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axs[i, 0].set_title(f"u0_origin[{i}]")
    axs[i, 0].set_xlabel("X-axis")
    axs[i, 0].set_ylabel("Y-axis")
    fig.colorbar(cax0, ax=axs[i, 0], orientation="vertical")

    # Plot u0_approx
    cax1 = axs[i, 1].imshow(u0_approx, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axs[i, 1].set_title(f"u0_approx[{i}]")
    axs[i, 1].set_xlabel("X-axis")
    axs[i, 1].set_ylabel("Y-axis")
    fig.colorbar(cax1, ax=axs[i, 1], orientation="vertical")

    # Plot u0_diff
    cax2 = axs[i, 2].imshow(u0_diff, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
    axs[i, 2].set_title(f"u0_diff[{i}]")
    axs[i, 2].set_xlabel("X-axis")
    axs[i, 2].set_ylabel("Y-axis")
    fig.colorbar(cax2, ax=axs[i, 2], orientation="vertical")

plt.tight_layout()
plt.savefig("tmp.png")
