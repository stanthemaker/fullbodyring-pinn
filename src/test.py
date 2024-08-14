import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from collections import OrderedDict

# data = np.load("/home/stan/data/pinn/data/0726_forward_417.npz")["data"]
# indices = [200, 250, 300, 350]
# slices = data[indices]
# print(slices.shape)
# np.savez_compressed(
#     "/home/stan/data/pinn/data/0726_forward_417_4slices.npz", data=slices
# )





# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# layers = [3] + [30] * 5 + [1]
# model = DNN(layers).to(device)
# summary(model, (3,))
