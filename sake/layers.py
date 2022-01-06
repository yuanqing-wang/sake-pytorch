import numpy as np
import torch
import dgl
from typing import Callable, Union
import itertools
from .utils import PNA, Coloring, RBF, HardCutOff, ContinuousFilterConvolution, ConcatenationFilter
from .functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)

class SAKELayer(torch.nn.Module):
    epsilon = 1e-10
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
    ):
        super().__init__()

        self.edge_model = distance_filter(2*in_features, hidden_features)

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )

        self.residual = residual
        self.update_coordinate = update_coordinate

        if update_coordinate:
            self.coordinate_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, 1, bias=False),
            )

            torch.nn.init.xavier_uniform_(self.coordinate_mlp[2].weight, gain=0.001)

        self.attention = attention
        if attention is True:
            self.att_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, 1),
                torch.nn.Sigmoid(),
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

class DenseSAKELayer(SAKELayer):
    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, mask=None):
        # (batch_size, n, n, coefficients)
        coefficients = self.coefficients_mlp(h_e_mtx)

        # (batch_size, n, n, coefficients, 3)
        combinations = coefficients.unsqueeze(-1) * ((x_minus_xt / (x_minus_xt_norm ** 2.0 + self.epsilon)).unsqueeze(-2))

        if mask is not None:
            combinations = combinations * mask.unsqueeze(-1).unsqueeze(-1)

        # (batch_size, n, n, coefficients)
        combinations_sum = combinations.sum(dim=-3)
        combinations_norm = combinations_sum.pow(2).sum(-1)
        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * mask.unsqueeze(-1)
        h_e = h_e_mtx.sum(dim=-2)
        return h_e

    def node_model(self, h, h_e, h_combinations):
        out = torch.cat([h, h_e, h_combinations], dim=-1)
        out = self.node_mlp(out)
        if self.residual:
            out = h + out
        return out

    def coordinate_model(self, x, x_minus_xt, h_e_mtx):
        translation = x_minus_xt * self.coordinate_mlp(h_e_mtx)
        agg = translation.mean(dim=-2)
        x = x + agg
        return x

    def forward(self, h, x, mask=None):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt, epsilon=0.0).pow(2)
        h_cat_ht = get_h_cat_h(h)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        h_combinations = self.spatial_attention(h_e_mtx, x_minus_xt, x_minus_xt_norm, mask=mask)
        h = self.node_model(h, h_e, h_combinations)
        if self.update_coordinate:
            x = self.coordinate_model(x, x_minus_xt, h_e_mtx)
        return h, x
