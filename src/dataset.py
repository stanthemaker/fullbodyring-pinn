# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from tqdm import tqdm


class wavDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path
        # self.transform = train_tfm
        wave = np.load(self.path)
        self.t, self.h, self.w = wave.shape
        wave = torch.from_numpy(wave.flatten()).float()
        self.wave = torch.nn.functional.normalize(wave, dim=0)

        indices = np.arange(self.t * self.w * self.h)
        t = indices // (self.w * self.h)
        r = indices % (self.w * self.h)
        x = r // self.w
        y = r % self.w

        coords = np.stack((t, x, y), axis=1)
        self.coords = torch.from_numpy(coords).float()

        self.rms = torch.sqrt(torch.mean(self.wave**2)).item()

    def __len__(self):
        return len(self.wave)

    def __getitem__(self, idx):

        return self.coords[idx], self.wave[idx]


# wav = wavDataset("/home/stan/data/pinn/slice.npy")
# for i in tqdm(range(wav.__len__())):
#    print(wav.__getitem__(i)[0])
# print(wav.__getitem__(250 * 500**2 + 200 * 500 + 200))
# data = np.load("/home/stan/data/pinn/downsampled_data.npy")
# print(data[250][200][200])
