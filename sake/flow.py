import abc
import torch
from .layers import DenseSAKELayer
from .functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)
import math
from typing import Union, Callable

class HamiltonianFlowLayer(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def f_forward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def f_backward(self, x, v, *args, **kwargs):
        raise NotImplementedError

class HamiltonianFlowModel(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def f_forward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def f_backward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    #
    # @abc.abstractmethod
    # def loss(self, z, *args, **kwargs):
    #     raise NotImplementedError

class SAKEFlowLayer(DenseSAKELayer, HamiltonianFlowLayer):
    steps = 3
    def __init__(self, *args, **kwargs):
        kwargs['update_coordinate'] = True
        kwargs['velocity'] = True
        super().__init__(*args, **kwargs)

        # if update_coordinate:
        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*self.in_features, self.hidden_features),
            self.activation,
            torch.nn.Linear(self.hidden_features, 1, bias=False),
            torch.nn.Tanh(),
        )

        self.radial_expansion_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_features + 1, self.hidden_features),
            self.activation,
            torch.nn.Linear(self.hidden_features, 1, bias=False),
        )

    def velocity_model(self, h, v):
        m = self.velocity_mlp(h).clamp(-10, 10)
        v = m.exp() * v
        log_det = m.sum((-1, -2)) * v.shape[-1]
        return v, log_det

    def invert_velocity_model(self, h, v):
        m = self.velocity_mlp(h).clamp(-10, 10)
        v = (-m).exp() * v
        log_det = m.sum((-1, -2)) * v.shape[-1]
        return v, log_det

    def radial_expansion_model(self, h, x, v):
        m = self.radial_expansion_mlp(
            torch.cat([h, v.pow(2).sum(dim=-1, keepdim=True)], dim=-1)
        ).clamp(-10, 10)
        x = m.exp() * x
        log_det = m.sum((-1, -2)) * x.shape[-1]
        return x, log_det

    def invert_radial_expansion_model(self, h, x, v):
        m = self.radial_expansion_mlp(
            torch.cat([h, v.pow(2).sum(dim=-1, keepdim=True)], dim=-1)
        ).clamp(-10, 10)
        x = (-m).exp() * x
        log_det = m.sum((-1, -2)) * x.shape[-1]
        return x, log_det

    def mp(self, h, x):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_h(h)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_combinations = self.spatial_attention(h_e_mtx, x_minus_xt, x_minus_xt_norm)
        combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx)
        h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_mtx)
        h = self.node_model(h, h_e, h_combinations)
        return h

    def f_backward(self, h, x, v):
        x, log_det_x = self.invert_radial_expansion_model(h, x, v)
        x = x - v
        for _ in range(self.steps):
            h = self.mp(h, x)
        h_cat_ht = get_h_cat_h(h)
        x_minus_xt = get_x_minus_xt(x)
        delta_v = self.coordinate_model(x, x_minus_xt, h_cat_ht)
        v = v - delta_v
        v, log_det_v = self.invert_velocity_model(h, v)
        log_det = log_det_x + log_det_v
        return x, v, log_det

    def f_forward(self, h, x, v):
        h0 = h
        for _ in range(self.steps):
            h = self.mp(h, x)
        h_cat_ht = get_h_cat_h(h)
        x_minus_xt = get_x_minus_xt(x)
        delta_v = self.coordinate_model(x, x_minus_xt, h_cat_ht)
        v, log_det_v = self.velocity_model(h, v)
        v = delta_v + v
        x = x + v
        x, log_det_x = self.radial_expansion_model(h0, x, v)
        log_det = log_det_x + log_det_v
        return x, v, log_det

class SAKEFlowModel(HamiltonianFlowModel):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            depth: int=4,
            activation: Callable=torch.nn.SiLU(),
        ):
        super().__init__()
        self.depth = depth
        self.embedding_in = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                SAKEFlowLayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                )
            )

    def f_forward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det = 0.0
        for layer in self.layers:
            x, v, log_det = layer.f_forward(h, x, v)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def f_backward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det = 0.0
        for layer in self.layers[::-1]:
            x, v, log_det = layer.f_backward(h, x, v)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def nll_backward(self, h, x, v, x_prior, v_prior):
        x, v, sum_log_det = self.f_backward(h, x, v)
        nll_x = -x_prior.log_prob(x).mean()
        nll_v = -v_prior.log_prob(v).mean()
        print(nll_x)
        return nll_x + nll_v + sum_log_det.mean()

class CenteredGaussian(torch.distributions.Normal):
    def __init__(self):
        super().__init__(loc=0.0, scale=1.0)
        self.device = "cpu"

    def to(self, device):
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        self.device = device
        return self

    def cuda(self):
        return self.to("cuda:0")

    def cpu(self):
        return self.to("cpu")

    def log_prob(self, value):
        N = value.shape[-2]
        D = value.shape[-1]
        degrees_of_freedom = (N-1) * D
        r2 = value.pow(2).flatten(-2, -1).sum(dim=-1)
        log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2*math.pi)
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, *args, **kwargs):
        x = super().sample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x

    def rsample(self, *args, **kwargs):
        x = super().rsample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x
