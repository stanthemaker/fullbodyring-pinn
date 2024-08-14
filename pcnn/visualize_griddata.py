import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# Load the coordinates and values from your files
grid_path = "/home/stan/data/pinn/pcnn/wavefields/wavefield_grid_for_dumps.txt"
U0_path = "/home/stan/data/pinn/pcnn/wavefields/wavefield0004500_01.txt"
coordinates = np.loadtxt(grid_path)

rows_to_shift = 20
cols = 500
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

X_spec = np.loadtxt(grid_path)
print("X_spec:", X_spec.shape)
X_spec = (
    X_spec / xz_scl
)  # specfem works with meters unit so we need to convert them to Km
X_spec[:, 0:1] = X_spec[:, 0:1]  # scaling the spatial domain
X_spec[:, 1:2] = X_spec[:, 1:2]  # scaling the spatial domain
xz_spec = np.concatenate((X_spec[:, 0:1], X_spec[:, 1:2]), axis=1)
U0 = np.loadtxt(U0_path)

n_eval = 100
x_eval = np.linspace(xmin, xmax, n_eval)
z_eval = np.linspace(zmin, zmax, n_eval)
x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
x_eval = x_eval_mesh.reshape(-1, 1)
z_eval = z_eval_mesh.reshape(-1, 1)
xz_eval = np.concatenate((x_eval, z_eval), axis=1)
u_eval1_0 = interpolate.griddata(xz_spec, U0, xz_eval, fill_value=0.0)  # [1600, 2]

print(xz_spec.shape, xz_eval.shape)  # (250000, 2) (10000, 2)
# print(xz_spec.shape, xz_eval.shape)  # (250000, 2) (10000, 2)
# u_eval1_0 = u_eval1_0.reshape(-1, 1)  # Reshape to (10000, 1) for concatenation
# wavefield_desc = np.concatenate((xz_eval, u_eval1_0), axis=1)
# np.savez_compressed("wave.npz", data=wavefield_desc)
half_size = int(u_eval1_0.size / 2)
u_eval1_0[:half_size] = u_eval1_0[half_size:]
u_eval1_0[half_size:] = 0

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axs[0].scatter(xz_spec[:, 0], xz_spec[:, 1], c=U0, cmap="viridis")
axs[0].set_title("xz_spec vs U0")
axs[0].set_xlabel("x")
axs[0].set_ylabel("z")
fig.colorbar(sc1, ax=axs[0])

# Subplot 2: xz_eval vs u_eval1_0
sc2 = axs[1].scatter(xz_eval[:, 0], xz_eval[:, 1], c=u_eval1_0, cmap="viridis")
axs[1].set_title("xz_eval vs u_eval1_0")
axs[1].set_xlabel("x")
axs[1].set_ylabel("z")
fig.colorbar(sc2, ax=axs[1])

plt.tight_layout()
plt.savefig("tmp.png")
# plt.show()
