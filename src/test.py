import numpy as np
import torch

# data = np.random.rand(100, 100, 100)
# np.save("test.npy", data)
data = np.load("/home/stan/data/pinn/output/0721-1258_0720_normdata_FNN_inference.npz")["data"]
print(data)
# np.savez_compressed("/home/stan/data/pinn/downsampled_data.npz", data=data)
