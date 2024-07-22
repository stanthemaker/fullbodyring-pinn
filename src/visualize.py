import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", type=str, default=None, help="npz file to load and visualize"
)
parser.add_argument("--num", "-n", type=int, default=0, help="slice index")
args = parser.parse_args()

if args.file.endswith(".npz"):
    data = np.load(args.file)["data"]
elif args.file.endswith(".npy"):
    data = np.load(args.file)

slice = data[args.num, :, :]
# w, h = slice.shape
# slice = torch.from_numpy(slice).float()
# slice = torch.nn.functional.normalize(slice.flatten(), dim=0).numpy().reshape(w, h)
# np.savez_compressed("slice.npz", data=slice)

plt.figure(figsize=(10, 8))
plt.imshow(slice, cmap="seismic")
plt.colorbar()
plt.title(f"Slice at index {args.num}")
plt.xlabel("Width")
plt.ylabel("Height")
plt.savefig("tmp.png")
