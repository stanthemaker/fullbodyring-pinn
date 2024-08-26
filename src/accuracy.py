import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import colors

analy_sol = loadmat("/home/stan/data/pinn/data/0815_homo.mat")
# pinn_sol = np.load("/home/stan/data/pinn/pcnn/3hz_homo/output.npz")["data"]

t0 = 285
t_end = 950

analy_sol = analy_sol["sr"].reshape(500, 500, -1)
analy_sol = np.transpose(analy_sol, (2, 0, 1))
# pinn_sol = pinn_sol.reshape(666, -1).reshape(-1, 100, 100)
print(analy_sol.shape)
np.savez_compressed("/home/stan/data/pinn/data/0815_homo.npz", data=analy_sol)
# print(pinn_sol.shape)

# data = analy_sol[285]
# print(np.min(data))
# plt.figure(figsize=(8, 6))
# cmap = plt.get_cmap("seismic")
# norm = colors.TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))
# plt.imshow(data, cmap=cmap, norm=norm, aspect="auto")
# plt.colorbar(label="Value")
# plt.title(f"Analy_sol")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.savefig("tmp.png")

# plt.figure(figsize=(12, 6))
# data = analy_sol[285]
# print(np.min(data))

# norm = colors.TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))
# plt.imshow(data, cmap=cmap, norm=norm, aspect="auto")
# plt.colorbar(label="Value")
# plt.title(f"Analy_sol")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.subplot(1, 2, 2)
# data = pinn_sol[0]
# print(np.min(data))
# norm = colors.TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=np.max(data))
# plt.imshow(data, cmap=cmap, norm=norm, aspect="auto")
# plt.colorbar(label="Value")
# plt.title(f"Pinn_sol")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.savefig("tmp.png")
