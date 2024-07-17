# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import torchvision.transforms as T

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# train_tfm = T.Compose(
#     [
#         # transforms.RandomResizedCrop((64, 64), (0.8, 1.25), (0.8, 1.25)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         # T.ToTensor(),
#         # transforms.Normalize(mean, std),
#     ]
# )


class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class wavDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path
        # self.transform = train_tfm
        self.wave = np.load(self.path)

    def __len__(self):
        return self.wave.size

    def __getitem__(self, idx):
        time, height, width = self.wave.shape
        t = idx // (height * width)
        x = (idx % (height * width)) // width
        y = idx % width

        input = torch.tensor([t, x, y]).float()
        target = torch.tensor(self.wave[t, x, y])

        return (input, target)


# wav = wavDataset("/home/stan/data/pinn/test.npy")
# input, target = wav.__getitem__(100)
# print(type(input))

# wav = np.load("/home/stan/data/pinn/wave_data.npy")
