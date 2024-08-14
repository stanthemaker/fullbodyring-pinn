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
print(np.std(slice))
# min_val = np.min(slice)
# max_val = np.max(slice)
# slice = 255 * (slice - min_val) / (max_val - min_val)
# slice = slice.astype(np.uint8)


plt.figure(figsize=(10, 8))
plt.imshow(slice, cmap="seismic", vmin=0, vmax=255)
plt.colorbar()
plt.title(f"Slice at index {args.num}")
plt.xlabel("Width")
plt.ylabel("Height")
plt.savefig("tmp.png")

# slice = np.expand_dims(slice, axis=0)
# print(slice.shape)
# # w, h = slice.shape
# # slice = torch.from_numpy(slice).float()
# # slice = torch.nn.functional.normalize(slice.flatten(), dim=0).numpy().reshape(w, h)
# np.savez_compressed("slice_norm255.npz", data=slice)
