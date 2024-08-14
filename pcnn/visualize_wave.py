import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# Load the coordinates and values from your files
grid_path = "/home/stan/data/pinn/pcnn/wavefields/wavefield_grid_for_dumps.txt"
U0_path = "/home/stan/data/pinn/pcnn/wavefields/wavefield0004500_01.txt"
wave_path = "/home/stan/data/pinn/data/0806_homo.npz"

wav_data = np.load(wave_path)["data"]
# coord = data[:, :2]
# val = data[:, 2]
# val[:5000] = val[5000:]
# val[5000:] = 0
# print(data.shape, coord.shape)

plt.figure(figsize=(8, 6))
plt.imshow(wav_data[100].T, cmap="seismic", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Value")
plt.title("2D Data with Seismic Colormap")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
plt.savefig("tmp.png")
