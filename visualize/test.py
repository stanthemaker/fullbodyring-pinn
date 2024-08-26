import numpy as np

homo = np.load("/Users/wengweishan/Documents/Caltech/pinn/0821_homo_analyt.npz")[
    "data"
][:1200]
inhomo = np.load("/Users/wengweishan/Documents/Caltech/pinn/0821_1disk_analyt.npz")[
    "data"
]
diff = inhomo - homo
np.savez_compressed("0821_diff.npz", data=diff)
np.savez_compressed("tmp.npz", data=homo)
print(homo.shape, inhomo.shape)
