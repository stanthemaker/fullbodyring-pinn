import numpy as np
import torch

data = np.random.rand(100, 100, 100)
np.save("test.npy", data)
# data = np.load("/home/stan/data/pinn/downsampled_data.npy")
# np.savez_compressed("/home/stan/data/pinn/downsampled_data.npz", data=data)
