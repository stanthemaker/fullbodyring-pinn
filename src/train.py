# Import necessary  standard packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import random
import os
import json

# self-defined modules
from dataset import wavDataset
from net import FNN, STMsFFN

# This is for the progress bar.
from tqdm import tqdm
from datetime import datetime


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
    n_epochs = config["nepochs"]
    lr = config["lr"]
    beta1 = config["beta1"]
    ckpt_dir = config["ckpt_dir"]
    data_path = config["data_path"]
    log_dir = config["log_dir"]
    NN_type = config["NN_type"]

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed = 1314520
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name
    log_file = os.path.join(log_dir, f"{train_name}.log")
    ckpt_file = os.path.join(ckpt_dir, f"{train_name}.pt")

    device = torch.device(cuda if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    dataset = wavDataset(path=data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    if NN_type == "STMsFNN":
        model = STMsFFN().to(device)
    else:
        model = FNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    min_loss = np.inf

    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        min_loss = state_dict["min_loss"]
        print(f"model loaded at {model_path}")

    gt_data = np.load(data_path)
    rms_gt = np.sqrt(np.mean(gt_data**2))
    print(f"ground truth RMS: {rms_gt}")

    loss_fn = nn.MSELoss()

    print("Starting Training Loop...")
    print("-" * 40)
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloader)
        progress_bar.set_description(f"Epoch {epoch+1}")
        mse_losses = []
        rms_losses = []
        for i, (input, target) in enumerate(progress_bar):
            input = input.to(device)
            target = torch.unsqueeze(target, dim=1).to(device)

            pred = model(input)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_losses.append(loss.item())
            rms_losses.append(np.sqrt(loss.item()))

        # # ------------------- an epoch finish ---------------------#
        mseloss = sum(mse_losses) / len(mse_losses)
        Rrmsloss = sum(rms_losses) / len(rms_losses) / rms_gt

        with open(log_file, "a") as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] mseloss = {mseloss:.4f}, rmseloss = {Rrmsloss:.4f}\n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] mseloss = {mseloss:.4f}, rmseloss = {Rrmsloss:.4f}\n"
            )

        if mseloss < min_loss:
            min_loss = mseloss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "min_loss": min_loss,
                },
                ckpt_file,
            )


if __name__ == "__main__":
    args = get_args()
    main(args.config, args.model, args.cuda, args.save)
