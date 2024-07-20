import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.autograd as autograd
from torch.autograd import Variable
from torchsummary import summary

# Define the Generator Network


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


import torch.nn as nn
import torch.nn.functional as F


class STMsFFN(nn.Module):
    def __init__(
        self,
        layer_sizes=[2] + [100] * 3 + [1],
        activation="tanh",
        kernel_initializer="Glorot uniform",
        sigmas_x=[1],
        sigmas_t=[1, 10],
        dropout_rate=0,
    ):
        super(STMsFFN, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = getattr(F, activation)
        self.dropout_rate = dropout_rate
        self.sigmas_x = sigmas_x
        self.sigmas_t = sigmas_t

        # Fourier feature parameters
        self.b = []
        for sigma in sigmas_x:
            b = nn.Parameter(
                torch.randn(layer_sizes[0] - 1, layer_sizes[1] // 2) * sigma,
                requires_grad=False,
            )
            self.register_parameter("b_x_" + str(sigma).replace(".", "_"), b)
            # print(f"b{b.shape}:{b}")
            self.b.append(b)

        for sigma in sigmas_t:
            b = nn.Parameter(
                torch.randn(1, layer_sizes[1] // 2) * sigma, requires_grad=False
            )
            self.register_parameter("b_t_" + str(sigma).replace(".", "_"), b)
            # print(f"b{b.shape}:{b}")
            self.b.append(b)

        # Fully-connected layers
        self.fcs = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.fcs.append(nn.Linear(layer_sizes[i + 1], layer_sizes[i + 2]))

        self.final_fc = nn.Linear(
            layer_sizes[-2] * (len(sigmas_x) * len(sigmas_t)), layer_sizes[-1]
        )

    def fourier_feature_forward(self, x, b):
        x_proj = torch.matmul(x, b)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fully_connected_forward(self, x):
        for fc in self.fcs:
            x = self.activation(fc(x))
        return x

    def forward(self, x):
        yb_x = [
            self.fourier_feature_forward(x[:, :-1], self.b[i])
            for i in range(len(self.sigmas_x))
        ]
        yb_t = [
            self.fourier_feature_forward(x[:, -1:], self.b[len(self.sigmas_x) + i])
            for i in range(len(self.sigmas_t))
        ]

        y_x = [self.fully_connected_forward(_yb) for _yb in yb_x]
        y_t = [self.fully_connected_forward(_yb) for _yb in yb_t]

        y = [y_x[i] * y_t[j] for i in range(len(y_x)) for j in range(len(y_t))]

        y = torch.cat(y, dim=1)
        y = self.final_fc(y)

        return y

    def fourier_feature_forward(self, x, b):
        x_proj = torch.matmul(x, b)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fully_connected_forward(self, x):
        for fc in self.fcs:
            x = self.activation(fc(x))
        return x


class FNN(nn.Module):
    """
    Input shape: (batch, 3)
    Output shape: (batch, 1)
    """

    def __init__(self, input_dim=3, output_dim=1):
        super(FNN, self).__init__()

        # Input layer: (batch, 3)
        self.l1 = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh())

        # Hidden layer

        self.l2 = nn.Sequential(
            nn.Linear(64, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 64),
            nn.Tanh(),
        )

        self.l3 = nn.Sequential(nn.Linear(64, output_dim))

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        return y


class smallFNN(nn.Module):
    """
    Input shape: (batch, 3)
    Output shape: (batch, 1)
    """

    def __init__(self, input_dim=3, output_dim=1):
        super(smallFNN, self).__init__()

        # Input layer: (batch, 3)
        self.l1 = nn.Sequential(nn.Linear(input_dim, 10), nn.Tanh())

        # Hidden layer
        self.l2 = nn.Sequential(nn.Linear(10, 10), nn.Tanh())

        self.l3 = nn.Sequential(nn.Linear(10, output_dim))

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        return y


# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# model = FNN().to(device)
# input = torch.tensor(np.random.rand(10, 3), dtype=torch.float32).to(device)
# summary(model, (3,))
# model(input)
# print(model)
