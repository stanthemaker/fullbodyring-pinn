import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.autograd as autograd
from torch.autograd import Variable

# Define the Generator Network


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


# ----------------------- these are models from ML ----------------------------
class FNN_Network(nn.Module):
    """
    Input shape: (batch, 3)
    Output shape: (batch, 1)
    """

    def __init__(self, input_dim=3, hidden_dim=10, output_dim=1):
        super(FNN_Network, self).__init__()

        # Input layer: (batch, 3)
        self.l1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Hidden layer
        self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # Output layer: (batch, 1)
        self.l3 = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        return y
