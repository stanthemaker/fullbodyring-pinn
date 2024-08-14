# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


class wavDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path
        self.wave = None
        if self.path.endswith(".npz"):
            self.wave = np.load(self.path)["data"].astype(np.float32)
        elif self.path.endswith(".npy"):
            self.wave = np.load(self.path).astype(np.float32)

        t, h, w = self.wave.shape
        # assume self.h = self.w
        x1 = np.linspace(0, 1, h + 1)[:-1]
        x1_t = np.linspace(0, 1, t + 1)[:-1]
        self.x_train = np.stack(np.meshgrid(x1, x1_t, x1), axis=-1)

        self.rms = np.sqrt(np.mean(self.wave**2))

    def __len__(self):
        return self.wave.shape[0]

    def __getitem__(self, idx):
        x = self.x_train[idx]
        x = torch.tensor(x, dtype=torch.float32)

        y = self.wave[idx]
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# wav = wavDataset("/home/stan/data/pinn/src/0726_forwardwave.npz")
# dataloader = DataLoader(
#     wav,
#     batch_size=5,
#     shuffle=False,
#     num_workers=1,
#     pin_memory=True,
# )
# wavs = []
# for i, (_, target) in enumerate(dataloader):
#     print(target)
#     wavs.append(target.numpy())
# wavs = np.concatenate(wavs, axis=0)
# print(wavs.reshape(1, 5, 5))
# print(wav.__getitem__(100))
# data = np.load("/home/stan/data/pinn/downsampled_data.npy")
# print(data[250][200][200])
