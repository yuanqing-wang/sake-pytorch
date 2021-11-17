import torch
import dgl
from typing import Union, Callable
from .layers import DenseSAKELayer, SparseSAKELayer

class DenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        activation: Callable=torch.nn.SiLU(),
        sum_readout: Union[None, Callable]=None,
        batch_norm: bool=False,
        *args, **kwargs,
    ):
        super(DenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.activation = activation
        self.depth = depth
        self.sum_readout = sum_readout
        self.batch_norm = batch_norm
        self.eq_layers = torch.nn.ModuleList()

        for idx in range(0, depth):
            self.eq_layers.append(
                DenseSAKELayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    *args, **kwargs,
                )
            )

    def forward(self, h, x):
        h = self.embedding_in(h)
        for idx, eq_layer in enumerate(self.eq_layers):
            h_ = h
            h, x = eq_layer(h, x)
            h = self.activation(h)
            h = h + h_

        h = self.embedding_out(h)
        if self.sum_readout is not None:
            h = h.sum(dim=-2)
            h = self.sum_readout(h)

        return h, x


class SparseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        activation: Callable=torch.nn.SiLU(),
        sum_readout: Union[None, Callable]=None,
        batch_norm: bool=False,
        *args, **kwargs,
    ):
        super(SparseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.activation = activation
        self.depth = depth
        self.sum_readout = sum_readout
        self.batch_norm = batch_norm
        self.eq_layers = torch.nn.ModuleList()

        for idx in range(0, depth):
            self.eq_layers.append(
                SparseSAKELayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    *args, **kwargs,
                )
            )

    def forward(self, g, h, x):
        h = self.embedding_in(h)
        for idx, eq_layer in enumerate(self.eq_layers):
            h_ = h
            h, x = eq_layer(g, h, x)
            h = self.activation(h)
            h = h + h_

        h = self.embedding_out(h)
        if self.sum_readout is not None:
            h = h.sum(dim=-2)
            h = self.sum_readout(h)

        return h, x
