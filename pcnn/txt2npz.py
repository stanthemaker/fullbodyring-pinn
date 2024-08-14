import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# Paths
wavefields_path = "/home/stan/data/pinn/pcnn/wavefields"
dump_dir = "/home/stan/data/pinn/data"

# Load data
X_spec = np.loadtxt(wavefields_path + "/wavefield_grid_for_dumps.txt")
wave_filed_dir_list = sorted(os.listdir(wavefields_path))[:-1]
U0 = [np.loadtxt(wavefields_path + "/" + f) for f in wave_filed_dir_list]

# Scaling the spatial domain
xz_scl = 600
X_spec = X_spec / xz_scl
u_scl = max(abs(np.min(U0[0])), abs(np.max(U0[0])))
print("scale:", u_scl)
# Coordinates and values
x = X_spec[:, 0]
y = X_spec[:, 1]
for i, u in enumerate(U0):
    values = U0[i]
    grid_x, grid_y = np.mgrid[x.min() : x.max() : 500j, y.min() : y.max() : 500j]
    U0[i] = griddata((x, y), values, (grid_x, grid_y), method="cubic") / u_scl
    U0[i] = U0[i].reshape(1, 500, 500)

# # Plot the result
plt.figure(figsize=(8, 6))
# plt.imshow(grid_z.T, extent=(x.min()*xz_scl, x.max()*xz_scl, y.min()*xz_scl, y.max()*xz_scl), origin='lower', cmap='seismic', alpha=1)
plt.imshow(U0[0].T, cmap="seismic", aspect="auto")
plt.colorbar(label="Value")
plt.title("Interpolated 2D Data")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.savefig("tmp.png")

# print("scale:", u_scl)
# for i, u in enumerate(U0):
#     U0[i] = u.reshape(1, 500, 500) / u_scl
U0 = np.concatenate(U0, axis=0)
print(U0.shape)
save_path = os.path.join(dump_dir, "origin4slice")
np.savez_compressed(save_path, data=U0)
