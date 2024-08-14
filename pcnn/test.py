import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.io import savemat
import scipy.interpolate as interpolate

# Load the prediction and ground truth data
pred = np.load("//home/stan/data/pinn/pcnn/3hz_0813/inference/pred.npz")["data"]
gt = np.load("/home/stan/data/pinn/data/0813_f3wav.npz")["data"]
# Define the region of interest
gt_idx = 410
pred_idx = 2
u_pred = pred[pred_idx].reshape(-1)
u_scl = max(abs(np.min(gt[285])), abs(np.max(gt[285])))

# Process the ground truth data
# for i in range(start, stop, 1):
#     u_gt = gt[i]
#     u_gt = zoom(u_gt, (100 / u_gt.shape[0], 100 / u_gt.shape[1]), order=1)
#     u_gt = u_gt.reshape(-1) / u_scl
#     print(i, np.sqrt(np.mean((u_gt - u_pred) ** 2)))

# exit()
# Scaling factors and boundaries
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

# Evaluation points
n_eval = 100
x_eval = np.linspace(xmin, xmax, n_eval)
z_eval = np.linspace(zmin, zmax, n_eval)
x_eval_mesh, z_eval_mesh = np.meshgrid(x_eval, z_eval)
x_eval = x_eval_mesh.reshape(-1, 1)
z_eval = z_eval_mesh.reshape(-1, 1)
xz_eval = np.concatenate((x_eval, z_eval), axis=1)  # [1600, 2]

x_0 = np.linspace(0, 1500, 500) / xz_scl
z_0 = np.linspace(0, 1500, 500) / xz_scl
x_0_mesh, z_0_mesh = np.meshgrid(x_0, z_0)
x_0 = x_0_mesh.reshape(-1, 1)
z_0 = z_0_mesh.reshape(-1, 1)
xz_0 = np.concatenate((x_0, z_0), axis=1)
values = gt[gt_idx].reshape(-1)
u_eval1_0 = interpolate.griddata(xz_0, values, xz_eval, fill_value=0.0)  # [1600, 2]
u_eval1 = u_eval1_0.reshape(100, 100) / u_scl


# u_eval1 = u_eval1_0.reshape(-1, 1) / u_scl
# u_eval2_0 = interpolate.griddata(xz_spec, U0[1], xz_eval, fill_value=0.0)
# u_eval2 = u_eval2_0.reshape(-1, 1) / u_scl
# u_eval3_0 = interpolate.griddata(
#     xz_spec, U0[2], xz_eval, fill_value=0.0
# )  # Test data
# u_eval3 = u_eval3_0.reshape(-1, 1) / u_scl
# u_eval4_0 = interpolate.griddata(
#     xz_spec, U0[3], xz_eval, fill_value=0.0
# )  # Test data
# u_eval4 = u_eval4_0.reshape(-1, 1) / u_scl

# # Compute the difference between the predicted and ground truth data
u_gt = gt[gt_idx]
u_gt = zoom(u_gt, (100 / u_gt.shape[0], 100 / u_gt.shape[1]), order=1) / u_scl
# u_gt = u_gt.reshape(-1) / u_scl

u_pred = u_pred.reshape(100, 100)
_1d_pred = u_pred[:, 50]
_1d_gt_intpl = u_eval1[:, 50]
_1d_gt_zoom = u_gt[:, 50]
print(np.sqrt(np.mean(_1d_gt_intpl - _1d_gt_zoom) ** 2))
print(np.sqrt(np.mean(_1d_pred - _1d_gt_zoom) ** 2))
print(np.sqrt(np.mean(_1d_pred - _1d_gt_intpl) ** 2))

plt.figure(figsize=(8, 6))
plt.plot(_1d_gt_intpl, label="interpolate", color="blue", alpha=0.7)
plt.plot(_1d_gt_zoom, label="zoom", color="green", alpha=0.7)
plt.plot(_1d_pred, label="pred", color="red", alpha=0.7)
plt.xlabel("Z-axis Index")
plt.ylabel("Value")
plt.title(f"index: {gt_idx}")
plt.legend()
plt.grid(True)
plt.savefig("tmp.png")

# data_dict = {"pred": _1d_pred, "gt": _1d_gt}
# savemat("slice3.mat", data_dict)

# plt.figure(figsize=(8, 6))
# # plt.imshow(wav_data[160], cmap='seismic', aspect='auto', vmin = -1 , vmax =1)
# plt.imshow(u_pred, cmap="seismic", aspect="auto")
# plt.colorbar(label="Value")
# plt.title("2D Data with Seismic Colormap")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# # Plot the ground truth
# axs[0].scatter(
#     x_eval * xz_scl,
#     z_eval * xz_scl,
#     c=u_gt,
#     edgecolors="none",
#     cmap="seismic",
#     marker="o",
#     s=10,
#     vmin=-1,
#     vmax=1,
# )
# axs[0].set_title("Ground Truth (u_gt)")
# axs[0].axis("equal")
# axs[0].set_xticks([])
# axs[0].set_yticks([])

# # Plot the prediction
# axs[1].scatter(
#     x_eval * xz_scl,
#     z_eval * xz_scl,
#     c=u_pred,
#     edgecolors="none",
#     cmap="seismic",
#     marker="o",
#     s=10,
#     vmin=-1,
#     vmax=1,
# )
# axs[1].set_title("Prediction (u_pred)")
# axs[1].axis("equal")
# axs[1].set_xticks([])
# axs[1].set_yticks([])

# # Plot the difference
# sc = axs[2].scatter(
#     x_eval * xz_scl,
#     z_eval * xz_scl,
#     c=u_diff,
#     edgecolors="none",
#     cmap="seismic",
#     marker="o",
#     s=10,
# )
# axs[2].set_title("Difference (u_pred - u_gt)")
# axs[2].axis("equal")
# axs[2].set_xticks([])
# axs[2].set_yticks([])

# # Add a colorbar to the difference plot
# fig.colorbar(sc, ax=axs[2], orientation="vertical")
