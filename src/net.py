import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.autograd as autograd
from torch.autograd import Variable
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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


class DNN_ReLU(nn.Module):
    def __init__(self, input_dim=3, hiddenlayer_dim=400, output_dim=1):
        super(DNN_ReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hiddenlayer_dim),
            nn.ReLU(),
            nn.Linear(hiddenlayer_dim, hiddenlayer_dim),
            nn.ReLU(),
            nn.Linear(hiddenlayer_dim, hiddenlayer_dim),
            nn.ReLU(),
            nn.Linear(hiddenlayer_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class DNN_Tanh(nn.Module):
    def __init__(self, input_dim=3, hiddenlayer_dim=256, output_dim=1):
        super(DNN_Tanh, self).__init__()
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


class DNN_paper(nn.Module):
    def __init__(self, layers=[3] + [30] * 5 + [1]):
        super(DNN_paper, self).__init__()

        self.depth = len(layers) - 1

        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


model_classes = {
    "STMsFNN": STMsFFN,
    "DNN_Tanh": DNN_Tanh,
    "DNN_ReLU": DNN_ReLU,
    "DNN_paper": DNN_paper,
}

# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# input = torch.tensor(np.random.rand(10, 3), dtype=torch.float32).to(device)
# model = FNN().to(device)
# model(input)
# print(model)
