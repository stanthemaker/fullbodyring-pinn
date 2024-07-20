# Import necessary  standard packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import os
import json
from tqdm import tqdm

# self-defined modules
from net import FNN, STMsFFN

# usage: python3 inference.py --model ../ckpt/0718-1642_fcnn.pt --output_dir ../output/ --cuda cuda:1

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default=None, help="Model path to load")
parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    help="output path to store predicted data",
)
parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")

args = parser.parse_args()
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")


grid_size = 500
t = np.arange(grid_size)  # 0 to 499
x = np.arange(grid_size)
y = np.arange(grid_size)

tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
inputs = np.stack([tt, xx, yy], axis=-1).reshape(-1, 3)
inputs = torch.from_numpy(inputs).float()

model = FNN().to(device)
state_dict = torch.load(args.model)
model.load_state_dict(state_dict["model"])
model.eval()
print("model loaded")

batch_size = 10000
outputs = []

with torch.no_grad():
    for i in tqdm(range(0, inputs.shape[0], batch_size)):

        batch_inputs = inputs[i : i + batch_size].to(device)
        batch_outputs = model(batch_inputs).detach().cpu().numpy()
        outputs.append(batch_outputs)

# Concatenate all the outputs
print("output done")
outputs = np.concatenate(outputs, axis=0)
print(outputs.shape)
outputs = np.reshape(outputs, (grid_size, grid_size, grid_size))
print(outputs.shape)

savefile = os.path.join(args.output_dir, "predicted_data.npz")
np.savez_compressed(savefile, data=outputs)
print("Predicted data saved")
