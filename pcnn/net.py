from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import sobol_sequence
from torch.optim.lr_scheduler import StepLR
from functorch import jacrev, vmap, make_functional, grad, vjp
import torch.autograd.functional as F
import timeit
from tqdm import tqdm
from utils import plot_eval, bilinear_interpol


def initialize_weights(module):
    """starting from small initialized parameters"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


def save_checkpoint(model, save_dir):
    """save model and optimizer"""
    torch.save({"model_state_dict": model.state_dict()}, save_dir)
    print("Pretrained model saved!")


def load_checkpoint(model, save_dir):
    """load model and optimizer"""
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Pretrained model loaded!")

    return model


# the deep neural network
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
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


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(self, **kwargs):
        # Define layers
        self.log_file = kwargs.pop("log_file", "")
        self.map_file = kwargs.pop("map_file", "")  # sos map
        self.kernel_size = kwargs.pop("kernel_size", 200)
        self.fig_dir = kwargs.pop("fig_dir", "")
        self.ckpt_dir = kwargs.pop("ckpt_dir", "")
        self.inf_dir = kwargs.pop("inf_dir", "")
        self.device = kwargs.pop("device", "cuda:0")
        self.mode = kwargs.pop("mode", "train")

        lbfgs_iter = kwargs.pop("lbfgs_iter", 6000)
        model_path = kwargs.pop("model_path", "")
        layers = kwargs.pop("layers", [3] + [40] * 5 + [1])

        self.xz_scl = kwargs.pop("xz_scl", 0)
        self.time_pts = kwargs.pop("time_pts", None)
        self.xmax = kwargs.pop("xmax", None)
        self.xmin = kwargs.pop("xmin", None)
        self.zmax = kwargs.pop("zmax", None)
        self.zmin = kwargs.pop("zmin", None)
        self.xini_min = kwargs.pop("xini_min", None)
        self.xini_max = kwargs.pop("xini_max", None)

        self.X_evals = kwargs.pop("X_evals", None)
        self.p_evals = kwargs.pop("p_evals", None)
        self.x_eval = kwargs.pop("x_eval", None)
        self.z_eval = kwargs.pop("z_eval", None)

        device = self.device
        if not self.map_file == "":
            self.sos_map = np.load(self.map_file)["data"] / self.xz_scl
        # self.sos_map = torch.tensor(smap, dtype=torch.float64).to(device)
        # deep neural networks
        self.dnn = DNN(layers).to(device)

        print("model_path:", model_path)
        if model_path == "":
            self.dnn.apply(initialize_weights)
        else:
            self.dnn = load_checkpoint(self.dnn, model_path)

        # input data
        X_pde = kwargs.pop("X_pde", None)
        if X_pde is not None and X_pde.size > 0 and X_pde.any():
            self.x_pde = torch.tensor(
                X_pde[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.z_pde = torch.tensor(
                X_pde[:, 1:2], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.t_pde = torch.tensor(
                X_pde[:, 2:3], dtype=torch.float64, requires_grad=True
            ).to(device)

        ini_cond = kwargs.pop("ini_cond", None)
        if not ini_cond == None:
            X_ini1 = ini_cond.get("X_ini1")
            X_ini2 = ini_cond.get("X_ini2")
            p_ini1 = ini_cond.get("p_ini1")
            p_ini2 = ini_cond.get("p_ini2")
            X_ini1_s2 = ini_cond.get("X_ini1_s2")
            X_ini2_s2 = ini_cond.get("X_ini2_s2")
            p_ini1_s2 = ini_cond.get("p_ini1_s2")
            p_ini2_s2 = ini_cond.get("p_ini2_s2")

            self.x_ini1 = torch.tensor(
                X_ini1[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.z_ini1 = torch.tensor(
                X_ini1[:, 1:2], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.t_ini1 = torch.tensor(
                X_ini1[:, 2:3], dtype=torch.float64, requires_grad=True
            ).to(device)

            self.x_ini2 = torch.tensor(
                X_ini2[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.z_ini2 = torch.tensor(
                X_ini2[:, 1:2], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.t_ini2 = torch.tensor(
                X_ini2[:, 2:3], dtype=torch.float64, requires_grad=True
            ).to(device)

            self.p_ini1 = torch.tensor(
                p_ini1[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.p_ini2 = torch.tensor(
                p_ini2[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)

            self.x_ini1_s2 = torch.tensor(
                X_ini1_s2[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.z_ini1_s2 = torch.tensor(
                X_ini1_s2[:, 1:2], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.t_ini1_s2 = torch.tensor(
                X_ini1_s2[:, 2:3], dtype=torch.float64, requires_grad=True
            ).to(device)

            self.x_ini2_s2 = torch.tensor(
                X_ini2_s2[:, 0:1], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.z_ini2_s2 = torch.tensor(
                X_ini2_s2[:, 1:2], dtype=torch.float64, requires_grad=True
            ).to(device)
            self.t_ini2_s2 = torch.tensor(
                X_ini2_s2[:, 2:3], dtype=torch.float64, requires_grad=True
            ).to(device)

            self.p_ini1_s2 = torch.tensor(
                p_ini1_s2, dtype=torch.float64, requires_grad=True
            ).to(device)
            self.p_ini2_s2 = torch.tensor(
                p_ini2_s2, dtype=torch.float64, requires_grad=True
            ).to(device)

        # optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            max_iter=lbfgs_iter,
            max_eval=lbfgs_iter,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        self.lr_adam = 5.0e-3
        self.opt_adam = torch.optim.Adam(self.dnn.parameters(), lr=self.lr_adam)
        self.scheduler = StepLR(self.opt_adam, step_size=100, gamma=0.99)

        self.LBFGS_iter = 0
        self.adam_iter = 0

        self.K_ini1_log = []
        self.K_ini2_log = []
        self.K_pde_log = []

        self.loss_adam = []
        self.loss_ini_adam = []
        self.loss_pde_adam = []

        self.lambda_ini1_log = []
        self.lambda_ini2_log = []
        self.lambda_pde_log = []
        self.L2_error_log = []

    def net_u(self, x, z, t):
        u = self.dnn(torch.cat((x, z, t), dim=1))
        return u

    def get_sos(self, x, z):
        x_np = x.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        x_np = (
            (x_np - self.xmin) / (self.xmax - self.xmin) * (self.sos_map.shape[0] - 1)
        )
        z_np = (
            (z_np - self.zmin) / (self.zmax - self.zmin) * (self.sos_map.shape[0] - 1)
        )

        sos = bilinear_interpol(x_np, z_np, self.sos_map)
        return torch.tensor(sos, dtype=torch.float64).to(self.device)

    def calc_res_pde(self, x, z, t):

        u = self.net_u(x, z, t)
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_z = torch.autograd.grad(
            u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_zz = torch.autograd.grad(
            u_z,
            z,
            grad_outputs=torch.ones_like(u_z),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t,
            t,
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True,
        )[0]
        if self.map_file == "":
            res_pde = u_tt - (u_xx + u_zz)
        else:
            sos = self.get_sos(x, z)
            res_pde = u_tt / sos**2 - (u_xx + u_zz)

        return res_pde

    def compute_ini_ntk(self, x, z, t):
        fnet, params = make_functional(self.dnn)

        def fnet_single(params, x, z, t):
            X = torch.cat((x, z, t), dim=1)
            u = fnet(params, X)

            return u

        jac1 = jacrev(fnet_single)(params, x, z, t)  # 只算第一个参数的？
        jac1_flat = [j.flatten(2) for j in jac1]

        jac2 = jacrev(fnet_single)(params, x, z, t)
        jac2_flat = [j.flatten(2) for j in jac2]

        # full
        # einsum_expr = 'Naf,Mbf->NMab'
        # trace
        einsum_expr = "Naf,Maf->NM"

        # Compute J(x1) @ J(x2).T
        # TODO compute full or trace or diagnal
        result = torch.stack(
            [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1_flat, jac2_flat)]
        )
        result = result.sum(0).reshape(self.kernel_size, self.kernel_size)

        return result

    def compute_pde_ntk(self, x, z, t):
        fnet, params = make_functional(self.dnn)
        index = torch.ones((self.kernel_size, 1)).to(self.device)

        def fnet_x(params, x, z, t):
            X = torch.cat((x, z, t), dim=1)
            u = fnet(params, X)

            return u

        def calc_vjp_x(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_x

        def calc_vjp_z(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_z

        def calc_vjp_t(params, x, z, t):
            (vjp0, vjpfunc) = vjp(fnet_x, params, x, z, t)

            (_, vjp_x, vjp_z, vjp_t) = vjpfunc(index)

            return vjp_t

        def calc_vjp_xx(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_x, params, x, z, t)
            (_, vjp_xx, vjp_xz, vjp_xt) = vjpfunc(index)

            return vjp_xx

        def calc_vjp_zz(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_z, params, x, z, t)
            (_, vjp_zx, vjp_zz, vjp_zt) = vjpfunc(index)

            return vjp_zz

        def calc_vjp_tt(params, x, z, t):
            (vjp0, vjpfunc) = vjp(calc_vjp_t, params, x, z, t)
            (_, vjp_tx, vjp_tz, vjp_tt) = vjpfunc(index)

            return vjp_tt

        def calc_vjp_res(params, x, z, t):
            vjp_xx = calc_vjp_xx(params, x, z, t)
            vjp_zz = calc_vjp_zz(params, x, z, t)
            vjp_tt = calc_vjp_tt(params, x, z, t)

            res = vjp_tt - (vjp_xx + vjp_zz)

            return res

        jac1 = jacrev(calc_vjp_res)(params, x, z, t)
        jac1_flat = [j.flatten(2) for j in jac1]

        jac2 = jacrev(calc_vjp_res)(params, x, z, t)
        jac2_flat = [j.flatten(2) for j in jac2]
        # jac2_flat = jac1_flat

        # full
        # einsum_expr = 'Naf,Mbf->NMab'
        # trace
        einsum_expr = "Naf,Maf->NM"
        # diagonal
        # einsum_expr = 'Naf,Maf->NMa'

        # Compute J(x1) @ J(x2).T
        # TODO compute full or trace or diagnal
        result = torch.stack(
            [torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1_flat, jac2_flat)]
        )
        result = result.sum(0).reshape(self.kernel_size, self.kernel_size)

        return result

    def loss_func(self, IfIni):
        device = self.device
        mse_loss = nn.MSELoss()

        if IfIni:
            p_ini1_pred = self.net_u(self.x_ini1, self.z_ini1, self.t_ini1)
            p_ini2_pred = self.net_u(self.x_ini2, self.z_ini2, self.t_ini2)
            res_ini1 = p_ini1_pred - self.p_ini1
            res_ini2 = p_ini2_pred - self.p_ini2
            loss_ini1 = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device))
            loss_ini2 = mse_loss(res_ini2, torch.zeros_like(res_ini2).to(device))
            loss_pde = torch.tensor(0.0).to(device)
            loss = loss_ini1 + loss_ini2

        else:
            res_p_pred = self.calc_res_pde(self.x_pde, self.z_pde, self.t_pde)
            p_ini1_s2_pred = self.net_u(self.x_ini1_s2, self.z_ini1_s2, self.t_ini1_s2)
            p_ini2_s2_pred = self.net_u(self.x_ini2_s2, self.z_ini2_s2, self.t_ini2_s2)
            res_ini1 = p_ini1_s2_pred - self.p_ini1_s2
            res_ini2 = p_ini2_s2_pred - self.p_ini2_s2
            loss_ini1 = mse_loss(res_ini1, torch.zeros_like(res_ini1).to(device))
            loss_ini2 = mse_loss(res_ini2, torch.zeros_like(res_ini2).to(device))
            loss_pde = mse_loss(res_p_pred, torch.zeros_like(res_p_pred).to(device))
            loss = (
                self.lambda_ini1 * loss_ini1
                + self.lambda_ini2 * loss_ini2
                + self.lambda_pde * loss_pde
            )
        return loss, loss_ini1, loss_pde

    def train_adam(
        self, n_iters, calc_NTK=True, update_lambda=True, IfIni=True, loop_iter=0
    ):
        device = self.device
        kernel_size = self.kernel_size

        self.dnn.train()
        self.IfIni = IfIni
        self.loop_iter = loop_iter
        xmax = self.xmax
        xmin = self.xmin
        zmax = self.zmax
        zmin = self.zmin
        xini_min = self.xini_min
        xini_max = self.xini_max
        for epoch in range(n_iters):

            if calc_NTK & (epoch % 200 == 0):
                # print('calc NTK...')
                X_pde_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
                X_pde_NTK[:, 0] = X_pde_NTK[:, 0] * (xmax - xmin) + xmin
                X_pde_NTK[:, 1] = X_pde_NTK[:, 1] * (zmax - zmin) + zmin
                X_pde_NTK[:, 2] = X_pde_NTK[:, 2] * (
                    self.time_pts[-1] - self.time_pts[0]
                )

                X_ini1_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
                X_ini1_NTK[:, 0] = X_ini1_NTK[:, 0] * (xini_max - xini_min) + xini_min
                X_ini1_NTK[:, 1] = X_ini1_NTK[:, 1] * (xini_max - xini_min) + xini_min
                X_ini1_NTK[:, 2] = X_ini1_NTK[:, 2] * 0
                X_ini2_NTK = sobol_sequence.sample(kernel_size + 1, 3)[1:, :]
                X_ini2_NTK[:, 0] = X_ini2_NTK[:, 0] * (xini_max - xini_min) + xini_min
                X_ini2_NTK[:, 1] = X_ini2_NTK[:, 1] * (xini_max - xini_min) + xini_min
                X_ini2_NTK[:, 2] = X_ini2_NTK[:, 2] * 0 + (
                    self.time_pts[1] - self.time_pts[0]
                )

                self.x_pde_ntk = torch.tensor(
                    X_pde_NTK[:, 0:1], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.z_pde_ntk = torch.tensor(
                    X_pde_NTK[:, 1:2], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.t_pde_ntk = torch.tensor(
                    X_pde_NTK[:, 2:3], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.x_ini1_ntk = torch.tensor(
                    X_ini1_NTK[:, 0:1], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.z_ini1_ntk = torch.tensor(
                    X_ini1_NTK[:, 1:2], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.t_ini1_ntk = torch.tensor(
                    X_ini1_NTK[:, 2:3], dtype=torch.float64, requires_grad=True
                ).to(device)

                self.x_ini2_ntk = torch.tensor(
                    X_ini2_NTK[:, 0:1], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.z_ini2_ntk = torch.tensor(
                    X_ini2_NTK[:, 1:2], dtype=torch.float64, requires_grad=True
                ).to(device)
                self.t_ini2_ntk = torch.tensor(
                    X_ini2_NTK[:, 2:3], dtype=torch.float64, requires_grad=True
                ).to(device)

                self.K_ini1 = self.compute_ini_ntk(
                    self.x_ini1_ntk, self.z_ini1_ntk, self.t_ini1_ntk
                )
                self.K_ini2 = self.compute_ini_ntk(
                    self.x_ini2_ntk, self.z_ini2_ntk, self.t_ini2_ntk
                )
                self.K_pde = self.compute_pde_ntk(
                    self.x_pde_ntk, self.z_pde_ntk, self.t_pde_ntk
                )

                self.K_ini1_log.append(self.K_ini1.detach().cpu().numpy())
                self.K_ini2_log.append(self.K_ini2.detach().cpu().numpy())
                self.K_pde_log.append(self.K_pde.detach().cpu().numpy())

                # print('calc NTK end!')

            if calc_NTK & update_lambda & (epoch % 200 == 0):
                start_update_lambda = timeit.default_timer()
                # print('start update weights...')
                lambda_K_sum = (
                    torch.trace(self.K_ini1)
                    + torch.trace(self.K_ini2)
                    + torch.trace(self.K_pde)
                )

                lambda_ini1 = lambda_K_sum / torch.trace(self.K_ini1)
                lambda_ini2 = lambda_K_sum / torch.trace(self.K_ini2)
                lambda_pde = lambda_K_sum / torch.trace(self.K_pde)

                self.lambda_ini1 = torch.autograd.Variable(
                    lambda_ini1, requires_grad=True
                )
                self.lambda_ini2 = torch.autograd.Variable(
                    lambda_ini2, requires_grad=True
                )
                self.lambda_pde = torch.autograd.Variable(
                    lambda_pde, requires_grad=True
                )

                self.lambda_ini1_log.append(self.lambda_ini1.detach().cpu().numpy())
                self.lambda_ini2_log.append(self.lambda_ini2.detach().cpu().numpy())
                self.lambda_pde_log.append(self.lambda_pde.detach().cpu().numpy())

                stop_update_lambda = timeit.default_timer()
                print(
                    "Time: ",
                    stop_update_lambda - start_update_lambda,
                    "end of update weights...",
                )

            self.opt_adam.zero_grad()
            loss, loss_ini, loss_pde = self.loss_func(self.IfIni)
            loss.backward()
            self.opt_adam.step()

            self.scheduler.step()

            self.loss_adam.append(loss.detach().cpu().numpy())
            self.loss_ini_adam.append(loss_ini.detach().cpu().numpy())
            self.loss_pde_adam.append(loss_pde.detach().cpu().numpy())

            self.adam_iter += 1
            if self.adam_iter % 500 == 0:
                with open(self.log_file, "a") as f:
                    f.write(
                        f"loop_iter {self.loop_iter}, Adam Iter {self.adam_iter}, Loss: {loss.item():.4f}, loss_ini: {loss_ini.item():.6f}, loss_pde: {loss_pde.item():.4f}\n"
                    )
                    f.write(
                        f"lambda_ini1: {self.lambda_ini1}, lambda_pde: {self.lambda_pde}\n"
                    )

            if epoch % 2000 == 0:
                if IfIni:
                    fig_path = (
                        f"{self.fig_dir}/loop_{self.loop_iter}_ini_adam_{epoch}.png"
                    )
                    self.predict_eval(fig_path)
                else:
                    fig_path = f"{self.fig_dir}/loop_{self.loop_iter}_adam_{epoch}.png"
                    self.predict_eval(fig_path)

            if epoch % 10000 == 0:
                if IfIni:
                    save_model_path = f"{self.ckpt_dir}/loop_{self.loop_iter}_ini_adam_checkpoints_{epoch}.dump"
                    save_checkpoint(self.dnn, save_model_path)
                else:
                    save_model_path = f"{self.ckpt_dir}/loop_{self.loop_iter}_adam_checkpoints_{epoch}.dump"
                    save_checkpoint(self.dnn, save_model_path)

    def closure(self):
        self.optimizer.zero_grad()
        loss_LBFGS, loss_ini_LBFGS, loss_pde_LBFGS = self.loss_func(self.IfIni)
        loss_LBFGS.backward()
        self.loss_LBFGS.append(loss_LBFGS.detach().cpu().numpy())
        self.loss_ini_LBFGS.append(loss_ini_LBFGS.detach().cpu().numpy())
        self.loss_pde_LBFGS.append(loss_pde_LBFGS.detach().cpu().numpy())

        self.LBFGS_iter += 1
        if self.LBFGS_iter % 200 == 0:
            with open(self.log_file, "a") as f:
                f.write(
                    f"LBFGS: Iter {self.LBFGS_iter}, Loss: {loss_LBFGS.item():4f}, loss_ini: {loss_ini_LBFGS.item():.6f}, loss_pde: {loss_pde_LBFGS.item():.4f}\n"
                )

        if self.LBFGS_iter % 2000 == 0:
            if self.IfIni:
                fig_path = f"{self.fig_dir}/loop_{self.loop_iter}_ini_LBFGS_{self.LBFGS_iter}.png"
                self.predict_eval(fig_path)
                save_model_path = f"{self.ckpt_dir}/loop_{self.loop_iter}_ini_LBFGS_checkpoints_{self.LBFGS_iter}.dump"
                save_checkpoint(self.dnn, save_model_path)
            else:
                fig_path = (
                    f"{self.fig_dir}/loop_{self.loop_iter}_LBFGS_{self.LBFGS_iter}.png"
                )
                self.predict_eval(fig_path)
                save_model_path = f"{self.ckpt_dir}/loop_{self.loop_iter}_LBFGS_checkpoints_{self.LBFGS_iter}.dump"
                save_checkpoint(self.dnn, save_model_path)

        return loss_LBFGS

    def train_LBFGS(self):
        self.loss_LBFGS = []
        self.loss_ini_LBFGS = []
        self.loss_pde_LBFGS = []
        self.L2_error_LBFGS_log = []

        self.dnn.train()
        self.optimizer.step(self.closure)

    def predict(self, X_evalt):
        device = self.device
        x = torch.tensor(X_evalt[:, 0:1], dtype=torch.float64, requires_grad=True).to(
            device
        )
        z = torch.tensor(X_evalt[:, 1:2], dtype=torch.float64, requires_grad=True).to(
            device
        )
        t = torch.tensor(X_evalt[:, 2:3], dtype=torch.float64, requires_grad=True).to(
            device
        )

        self.dnn.eval()
        p = self.dnn(torch.cat((x, z, t), dim=1))

        return p

    def predict_eval(self, figname):

        P_PINN_pred = []
        P_diff_diff = []

        # Loop through the evaluation data
        for i in range(len(self.X_evals)):
            # Predict and convert to numpy
            p_eval = self.predict(self.X_evals[i])
            P_PINN = p_eval.detach().cpu().numpy()

            # Compute the difference
            P_diff = self.p_evals[i] - P_PINN

            # Append results to the lists
            P_PINN_pred.append(P_PINN)
            P_diff_diff.append(P_diff)

        kwargs = {
            "X_evals": self.X_evals,
            "P_PINN_pred": P_PINN_pred,
            "P_diff_diff": P_diff_diff,
            "p_evals": self.p_evals,
            "x_eval": self.x_eval,
            "z_eval": self.z_eval,
            "xz_scl": self.xz_scl,
            "savepath": figname,
        }
        plot_eval(**kwargs)

    def inference_field(self, X, savepath):
        output = []
        print("start inferencing ... ")
        for i in tqdm(range(len(X))):
            p_eval = self.predict(X[i])
            output.append(p_eval.detach().cpu().numpy())
        output = np.concatenate(output, axis=0)
        np.savez_compressed(savepath, data=output)


class Ultra_PINN(PhysicsInformedNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f = kwargs.pop("f", None)

    def loss_func(self):
        device = self.device
        mse_loss = nn.MSELoss()

        res_p_pred = self.calc_res_pde(self.x_pde, self.z_pde, self.t_pde)
        loss = mse_loss(
            res_p_pred, self.f(self.x_pde, self.z_pde, self.t_pde).to(device)
        )

        return loss

    def train_adam(self, n_iters):

        self.dnn.train()
        for epoch in range(n_iters):

            self.opt_adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.opt_adam.step()
            self.scheduler.step()

            self.loss_adam.append(loss.detach().cpu().numpy())

            self.adam_iter += 1
            if self.adam_iter % 500 == 0:
                with open(self.log_file, "a") as f:
                    f.write(f"Adam Iter {self.adam_iter}, Loss: {loss.item()}\n")

            if epoch % 2000 == 0:
                fig_path = f"{self.fig_dir}/adam_{epoch}.png"
                self.predict_eval(fig_path)

            if epoch % 10000 == 0:
                save_model_path = f"{self.ckpt_dir}/adam_checkpoints_{epoch}.dump"
                save_checkpoint(self.dnn, save_model_path)
