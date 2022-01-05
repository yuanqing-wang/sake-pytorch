import torch
from typing import Callable, Union
from ..utils import ConcatenationFilter
from ..functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)

class EGNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        update_coordinate: bool=False,
        residual: bool=False,
        activation: Union[None, Callable]=torch.nn.SiLU(),
        distance_filter: Callable=ConcatenationFilter,
        attention: bool=True,
    ):
        super().__init__()

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_features + 1, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
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

    def edge_model(self, h_cat_ht, x_minus_xt_norm):
        h_e_mtx = torch.cat(
            [
                h_cat_ht,
                x_minus_xt_norm,
            ],
            dim=-1
        )

        h_e_mtx = self.edge_mlp(h_e_mtx)

        if self.attention:
            att = self.att_mlp(h_e_mtx)
            h_e_mtx = att * h_e_mtx
        return h_e_mtx

    def mask_self(self, h_e_mtx):
        h_e_mtx = h_e_mtx * torch.eye(
            h_e_mtx.shape[-2],
            h_e_mtx.shape[-2],
            device=h_e_mtx.device,
        ).unsqueeze(-1)
        return h_e_mtx

    def aggregate(self, h_e_mtx, mask=None):
        h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * mask.unsqueeze(-1)
        h_e = h_e_mtx.sum(dim=-2)
        return h_e

    def node_model(self, h, h_e):
        out = torch.cat([h, h_e], dim=-1)
        out = self.node_mlp(out)
        if self.residual:
            out = h + out
        return out

    def coordinate_model(self, x, x_minus_xt, h_e_mtx):
        translation = x_minus_xt * self.coordinate_mlp(h_e_mtx)
        translation = self.mask_self(translation)
        agg = translation.mean(dim=-1)
        x = x + agg
        return x

    def forward(self, h, x, mask=None):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_h(h)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        if self.update_coordinate:
            x = self.coordinate_model(x, x_minus_xt, h_e_mtx)
        h = self.node_model(h, h_e)
        return h, x
