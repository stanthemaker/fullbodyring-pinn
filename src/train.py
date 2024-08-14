# Import necessary  standard packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import random
import os
import json

# self-defined modules
from dataset import wavDataset
from net import model_classes
from utils import plot_slices

# This is for the progress bar.
from tqdm import tqdm
from datetime import datetime

"""
    python3 train.py --config ./configs/[config file] --cuda cuda:1
"""


def get_args():
    parser = argparse.ArgumentParser(description=" NN data fitting")
    parser.add_argument("--config", "-j", type=str, help="config json file path")
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    parser.add_argument(
        "--save", "-s", type=int, default=1, help="whether to save model"
    )
    return parser.parse_args()


def main(config_path: str, model_path: str, cuda: str, to_save: int):
    with open(config_path, "r") as file:
        config = json.load(file)

    exp_name = config_path.split("/")[-1].split(".")[0]
    batch_size = config["batch_size"]
    n_epochs = config["nepochs"] + 1
    lr = config["lr"]
    ckpt_dir = config["ckpt_dir"]
    data_path = config["data_path"]
    log_dir = config["log_dir"]
    NN_type = config["NN_type"]

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed = 2318798
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name
    log_folder = os.path.join(log_dir, train_name)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    log_file = os.path.join(log_folder, f"{train_name}.log")
    ckpt_file = os.path.join(ckpt_dir, f"{train_name}.pt")

    device = torch.device(cuda if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    dataset = wavDataset(path=data_path)
    dataset_rms = dataset.rms
    print(f"dataset rms:{dataset_rms}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    model_class = model_classes.get(NN_type)
    model = model_class().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        print(f"model loaded at {model_path}")

    loss_fn = nn.MSELoss()
    print("Starting Training Loop...")
    print("-" * 40)
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloader)
        rec_indices = np.random.choice(progress_bar.total - 20, 4, replace=False) + 20
        progress_bar.set_description(f"Epoch {epoch+1}")
        mse_losses = []
        rec_preds, rec_gts = [], []

        for i, (input, target) in enumerate(progress_bar):
            input = input.to(device)
            target = target.to(device)
            pred = model(input).squeeze()

            loss = loss_fn(pred, target)
            # print(f"input shape: {input.shape}, target shape:{target.shape}")
            # print(f"output shape: {pred.shape}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_losses.append(loss.item())

            if i in rec_indices and epoch % 100 == 0:
                rec_preds.append(pred.detach().cpu().numpy()[0])
                rec_gts.append(target.detach().cpu().numpy()[0])

        # # ------------------- an epoch finish ---------------------#
        mseloss = sum(mse_losses) / len(mse_losses)
        Rrmsloss = np.sqrt(mseloss) / dataset_rms
        with open(log_file, "a") as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] mse = {mseloss:.7f}, relative rmse = {Rrmsloss:.4f}\n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] mse = {mseloss:.7f}, relative rmse = {Rrmsloss:.4f}\n"
            )
        if epoch % 100 == 0:
            rec_gts = np.stack(rec_gts, axis = 0)
            rec_preds = np.stack(rec_preds, axis = 0)
            plot_slices(rec_gts, rec_preds, os.path.join(log_folder, f"{epoch}.png"))
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_file,
            )


if __name__ == "__main__":
    args = get_args()
    main(args.config, args.model, args.cuda, args.save)
