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
from net import DNN_ReLU
from dataset import wavDataset

# usage: python3 inference.py --model ../ckpt/0718-1642_fcnn.pt --output_dir ../output/ --cuda cuda:1

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", "-m", type=str, default=None, help="Model path to load")
parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    default="/home/stan/data/pinn/output",
    help="output path to store predicted data",
)
parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
parser.add_argument(
    "--testset",
    "--t",
    type=str,
    default="/home/stan/data/pinn/data/0726_forward_417_4slices.npz",
    help="testing set",
)

args = parser.parse_args()
gt = np.load(args.testset)["data"]
print(gt.shape)
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")
testset = wavDataset(args.testset)
model = DNN_ReLU().to(device)
state_dict = torch.load(args.ckpt)
model.load_state_dict(state_dict["model"])
model.eval()

preds = []
with torch.no_grad():
    for i in tqdm(range(0, testset.__len__())):
        x, y = testset.__getitem__(i)
        x = x.to(device)
        pred = model(x).squeeze()
        preds.append(pred.detach().cpu().numpy())


preds = np.stack(preds, axis=0)
np.savez_compressed("predicted.npz", data = preds)
# predicted = np.zeros((t_size, x_size, y_size))
# predicted[coords[:, 0], coords[:, 1], coords[:, 2]] = outputs

# savefile = os.path.join(args.output_dir, f"{ckpt_name}_inference.npz")
# np.savez_compressed(savefile, data=predicted)
