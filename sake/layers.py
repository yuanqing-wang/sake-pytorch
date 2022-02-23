import numpy as np
import torch
import dgl
from typing import Callable, Union
import itertools
from .utils import (
    PNA, Coloring, RBF, HardCutOff,
    ContinuousFilterConvolution, ConcatenationFilter,
    ContinuousFilterConvolutionWithConcatenationRecurrent,
)
from .functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)


EPSILON = 1e-5
INF = 1e5

class SAKELayer(torch.nn.Module):
    epsilon = 1e-5
    inf = 1e5
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        update_coordinate: bool=False,
        residual: bool=True,
        activation: Union[None, Callable]=torch.nn.SiLU(),
        distance_filter: Callable=ConcatenationFilter,
        attention: bool=True,
        n_coefficients: int=32,
        n_heads: int=4,
        edge_features: int=0,
        velocity: bool=False,
        tanh: bool=False,
    ):
        super().__init__()

        self.edge_model = distance_filter(2*in_features+edge_features, hidden_features)

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_heads * hidden_features + hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )

        self.residual = residual
        self.edge_features = edge_features
        self.update_coordinate = update_coordinate
        self.velocity = velocity

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation

        # if update_coordinate:

        if tanh:
            self.coordinate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, 1, bias=False),
                torch.nn.Tanh(),
            )

            self.velocity_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, 1, bias=False),
                torch.nn.Tanh(),
            )

        else:

            self.coordinate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, 1, bias=False),
            )

            self.velocity_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, 1, bias=False),
            )

            torch.nn.init.xavier_uniform_(self.coordinate_mlp[2].weight, gain=0.001)
            torch.nn.init.xavier_uniform_(self.velocity_mlp[2].weight, gain=0.001)

        self.semantic_attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, n_heads),
            torch.nn.LeakyReLU(0.2),
        )

        self.coefficients_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, n_coefficients),
        )

        self.post_norm_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_coefficients, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.log_gamma = torch.nn.Parameter(torch.zeros(n_heads))

        self.n_heads = n_heads
        self.n_coefficients = n_coefficients

class DenseSAKELayer(SAKELayer):
    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, mask: Union[None, torch.Tensor]=None):
        # (batch_size, n, n, coefficients, 1)
        coefficients = self.coefficients_mlp(h_e_mtx).unsqueeze(-1)

        # (batch_size, n, n, 3)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5) ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = coefficients * x_minus_xt.unsqueeze(-2)

        if mask is not None:
            combinations = combinations * mask.unsqueeze(-1).unsqueeze(-1)

        # (batch_size, n, n, coefficients)
        combinations_sum = combinations.sum(dim=-3)
        combinations_norm = combinations_sum.pow(2).sum(-1)
        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations, combinations.mean(dim=(-2, -3))

    def aggregate(self, h_e_mtx, mask: Union[None, torch.Tensor]=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * mask.unsqueeze(-1)
        h_e = h_e_mtx.sum(dim=-2)
        return h_e

    def node_model(self, h, h_e, h_combinations):
        out = torch.cat([
                h,
                h_e,
                h_combinations,
            ],
            dim=-1)
        out = self.node_mlp(out)
        if self.residual:
            out = h + out
        return out

    def euclidean_attention(self, x_minus_xt_norm):
        # (batch_size, n, n, 1)
        _x_minus_xt_norm = x_minus_xt_norm + 1e5 * torch.eye(
            x_minus_xt_norm.shape[-2],
            x_minus_xt_norm.shape[-2],
            device=x_minus_xt_norm.device
        ).unsqueeze(-1)

        att = torch.nn.functional.softmin(
            _x_minus_xt_norm * self.log_gamma.exp(),
            dim=-2,
        )
        return att

    def semantic_attention(self, h_e_mtx):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        att = att - 1e5* torch.eye(
            att.shape[-2],
            att.shape[-2],
            device=att.device,
        ).unsqueeze(-1)
        att = torch.nn.functional.softmax(att, dim=-2)
        return att

    def combined_attention(self, x_minus_xt_norm, h_e_mtx):
        euclidean_attention = self.euclidean_attention(x_minus_xt_norm)
        semantic_attention = self.semantic_attention(h_e_mtx)
        combined_attention = (euclidean_attention * semantic_attention).softmax(dim=-2)
        return combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            v: Union[None, torch.Tensor]=None,
            mask: Union[None, torch.Tensor]=None,
            h_e_0: Union[None, torch.Tensor]=None,
        ):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_h(h)

        if self.edge_features > 0 and h_e_0 is not None:
            h_cat_ht = torch.cat([h_cat_ht, h_e_0], dim=-1)


        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_combinations, delta_v = self.spatial_attention(h_e_mtx, x_minus_xt, x_minus_xt_norm, mask=mask)

        if self.update_coordinate:
            # delta_v = self.coordinate_model(x, x_minus_xt, h_e_mtx)

            if v is not None and self.velocity:
                v = self.velocity_model(v, h)
            else:
                v = torch.zeros_like(x)

            v = delta_v + v
            x = x + v

        combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx)
        h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        h = self.node_model(h, h_e, h_combinations)
        return h, x, v
