import numpy as np
from SALib.sample import sobol_sequence
import matplotlib.pyplot as plt

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

time_span = 2.5
indices = [0, 100, 200, 300, 400]
eval_times = []
for i in range(len(indices)):
    eval_times.append((indices[i] / 1000) * time_span)
print("Eval times: ", eval_times)

mu_x, mu_z, mu_t = (
    xmax_spec / 2,
    zmax_spec / 2,
    0,
)
sigma_x, sigma_z, sigma_t = (
    xmax_spec / 10,
    zmax_spec / 10,
    time_span / 5,
)
n_samples = 5000
samples_3d = []
kernel_size = 200
X_pde_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
X_pde_NTK[:, 0] = X_pde_NTK[:, 0] * (xmax - xmin) + xmin
X_pde_NTK[:, 1] = X_pde_NTK[:, 1] * (zmax - zmin) + zmin
X_pde_NTK[:, 2] = X_pde_NTK[:, 2] * (
    eval_times[-1] - eval_times[0]
)

print(X_pde_NTK.shape)
# while len(samples_3d) < n_samples:
#     x = np.random.normal(loc=mu_x, scale=sigma_x)
#     z = np.random.normal(loc=mu_z, scale=sigma_z)
#     t = np.random.normal(loc=mu_t, scale=sigma_t)

#     if xmin <= x <= xmax and xmin <= z <= zmax and t_st <= t <= time_span:
#         samples_3d.append([x, z, t])
#     # if xmin <= x <= xmax and xmin <= z <= zmax:
#     # samples_2d.append([x, z])

# samples_3d = np.array(samples_3d)
# print(samples_3d.shape)


# def f_np(x, z, t):
#     f_central = 3
#     delay = 0.4
#     h = (2 * (np.pi * f_central * ((t + t_st) - delay))) * np.exp(
#         -((np.pi * f_central * ((t + t_st) - delay)) ** 2)
#     )
#     _lambda = (sos / xz_scl) / f_central
#     sigma = _lambda / 10
#     center = xmax_spec / 2
#     norm_factor = 1 / (2 * np.pi * sigma**2)
#     g = norm_factor * np.exp(
#         -((x - center) ** 2 / (2 * sigma**2) + (z - center) ** 2 / (2 * sigma**2))
#     )
#     return g * h


# ts = np.linspace(0, 0.5, 1000)
# print(ts)
# values_2d = np.array([f_np(1.25, 1.25, t) for t in ts])
# print(np.max(values_2d))

# Scatter plot for 2D visualization
# plt.scatter(samples_2d[:, 0], samples_2d[:, 1], c=values_2d, cmap="viridis", alpha=0.5)
# plt.colorbar(label="f(x,z)")
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.title("2D Visualization of f(x,z) at time = 0.5")
# plt.xlim(1.0, 1.5)
# plt.ylim(1.0, 1.5)
# plt.savefig("tmp.png")
