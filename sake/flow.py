import abc
import torch
from .layers import DenseSAKELayer
from .functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)
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
        )

    def velocity_model(self, v, h):
        m = self.velocity_mlp(h)
        v = m.exp() * v
        log_det = m.sum((-1, -2)) * v.shape[-1]
        return v, log_det

    def invert_velocity_model(self, v, h):
        m = self.velocity_mlp(h)
        v = (-m).exp() * v
        log_det = m.sum((-1, -2)) * v.shape[-1]
        return v, log_det

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
        x = x - v
        for _ in range(self.steps):
            h = self.mp(h, x)
        h_cat_ht = get_h_cat_h(h)
        x_minus_xt = get_x_minus_xt(x)
        delta_v = self.coordinate_model(x, x_minus_xt, h_cat_ht)
        v = v - delta_v
        v, log_det = self.invert_velocity_model(v, h)
        return x, v, log_det

    def f_forward(self, h, x, v):
        for _ in range(self.steps):
            h = self.mp(h, x)
        h_cat_ht = get_h_cat_h(h)
        x_minus_xt = get_x_minus_xt(x)
        delta_v = self.coordinate_model(x, x_minus_xt, h_cat_ht)
        v, log_det = self.velocity_model(v, h)
        v = delta_v + v
        x = x + v
        return x, v, log_det

class SAKEFlowModel(HamiltonianFlowModel):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int=6,
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
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def f_backward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det = 0.0
        for layer in self.layers:
            x, v, log_det = layer.f_backward(h, x, v)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det
