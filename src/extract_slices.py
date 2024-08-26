import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import argparse
import os


# slice1_idx = 150
# slice2_idx = 162
# slice3_idx = 258
# slice4_idx = 354
def get_args():
    parser = argparse.ArgumentParser(description="test alpha")
    # parser.add_argument("--alpha", "-a", type=float, default=0.0, help="alpha")
    parser.add_argument("--data", "-d", type=str, default="", help="datapath")
    parser.add_argument(
        "--time", "-t", type=float, default=1.5, help="time span for the whole data"
    )
    return parser.parse_args()


args = get_args()
slices = np.array([285, 318, 410, 510])
data_wave = np.load(args.data)["data"]
print(data_wave.shape)
savename = os.path.basename(args.data)[:-4] + "_4slices.npz"

save_path = os.path.join("/home/stan/data/pinn/data", savename)
p_new = np.empty((4, 500, 500))
p_scl = max(abs(np.min(data_wave[slices[0]])), abs(np.max(slices[0])))

for i, slice in enumerate(slices):

    p = data_wave[slice] / p_scl
    p_new[i] = p

data_dict = {
    "data": p_new,
    "index": slices,
    "n_slices": data_wave.shape[0],
    "time_span": args.time,
}

print(p_new.shape)
print(data_dict)
np.savez_compressed(save_path, **data_dict)
