import torch
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
import imageio
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim


def plot_slices(slices_gt, slices_pred, path):
    if slices_pred is not None:
        slices = np.concatenate((slices_gt, slices_pred), axis=0)
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize if needed
        axs = axs.ravel()
    else:
        slices = slices_gt
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))  # Adjust figsize if needed

    for i in range(slices.shape[0]):
        axs[i].imshow(slices[i], cmap="seismic")
        axs[i].axis("off")  # Optionally turn off the axis

    plt.tight_layout()
    plt.savefig(path)


seed = 1287632
torch.manual_seed(seed)
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

data = np.load("/home/stan/data/pinn/src/0726_forwardwave.npz")["data"]
min_val = np.min(data)
max_val = np.max(data)
average = np.mean(data)
print(min_val, max_val, average)
# data = (data - min_val) / (max_val - min_val)

# indices = np.random.choice(data.shape[0] - 150, 4, replace=False) + 100
indices = [200, 250, 300, 350]
slices = data[indices]

plot_slices(slices, slices_pred=None, path="/home/stan/data/pinn/output/FNN_3D/gt.png")

trainset = slices

RES = 417
# x1 = np.linspace(0, 1, RES//2+1)[:-1]
x1 = np.linspace(0, 1, RES + 1)[:-1]
x1_t = np.linspace(0, 1, 4 + 1)[:-1]
x_train = np.stack(np.meshgrid(x1, x1_t, x1), axis=-1)  # 3D
print("input spatio-temporal coordinate shape",x_train.shape)


class FNN(nn.Module):
    def __init__(self, input_dim=2, hiddenlayer_dim=256, output_dim=1):
        super(FNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hiddenlayer_dim),
            nn.Tanh(),
            nn.Linear(hiddenlayer_dim, hiddenlayer_dim),
            nn.Tanh(),
            nn.Linear(hiddenlayer_dim, hiddenlayer_dim),
            nn.Tanh(),
            nn.Linear(hiddenlayer_dim, hiddenlayer_dim),
            nn.Tanh(),
            nn.Linear(hiddenlayer_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
train_dict = {}
gt_rms = np.sqrt(np.mean(trainset**2))

def train_model(trainset, epochs, model_path):

    model = FNN(input_dim=3, hiddenlayer_dim=400, output_dim=1)
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        print(f"model loaded at {model_path}")

    model.train()

    for epoch in tqdm(range(epochs)):

        x = torch.tensor(x_train, dtype=torch.float32).to(device)
        y = torch.tensor(trainset, dtype=torch.float32).to(device)
        model = model.to(device)

        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = mse_loss_fn(output, y)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            output = output.detach().cpu().numpy()
            savepath = os.path.join(
                "/home/stan/data/pinn/output/FNN_3D/", f"{epoch}.png"
            )
            plot_slices(output, slices_pred=None, path=savepath)

            RMSE = np.sqrt(loss.item())
            print(
                f"Epoch {epoch+1}/{epochs}, MSELoss: {loss.item():.7f}, relative RMSE:{RMSE/gt_rms:.4f}"
            )

    train_dict["model"] = model.state_dict()
    train_dict["optimizer"] = optimizer.state_dict()
    train_dict["output"] = output


train_model(trainset, epochs=3000, model_path=None)
