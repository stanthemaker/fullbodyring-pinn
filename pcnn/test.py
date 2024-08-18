import numpy as np
from SALib.sample import sobol_sequence
import matplotlib.pyplot as plt

data = np.load("/home/stan/data/pinn/data/0815_homo_analyt.npz")["data"]

plt.imshow(data[0])
plt.savefig

