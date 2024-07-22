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
    default="/home/stan/data/pinn/output",
    help="output path to store predicted data",
)
parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")

args = parser.parse_args()
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")

t_size = 1
x_size = 500
y_size = 500
indices = np.arange(t_size * x_size * y_size)
t = indices // (x_size * y_size)
r = indices % (x_size * y_size)
x = r // x_size
y = r % x_size

coords = np.stack((t, x, y), axis=1)
inputs = torch.from_numpy(coords).float()

model = FNN().to(device)
state_dict = torch.load(args.model)
ckpt_name = args.model.split("/")[-1].split(".")[0]
model.load_state_dict(state_dict["model"])
model.eval()
print("model loaded")

batch_size = 10000
outputs = []

with torch.no_grad():
    # for i in tqdm(range(0, inputs.size()[0], batch_size)):
    for i in range(0, inputs.size()[0], batch_size):
        batch_inputs = inputs[i : i + batch_size].to(device)

        batch_outputs = model(batch_inputs).squeeze().detach().cpu().numpy()
        # print(batch_inputs, batch_outputs)
        outputs.append(batch_outputs)

outputs = np.concatenate(outputs, axis=0)
predicted = np.zeros((t_size, x_size, y_size))
predicted[coords[:, 0], coords[:, 1], coords[:, 2]] = outputs

savefile = os.path.join(args.output_dir, f"{ckpt_name}_inference.npz")
np.savez_compressed(savefile, data=predicted)
print("Predicted data saved")
