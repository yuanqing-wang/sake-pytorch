import torch
import dgl
from typing import Union, Callable, List
from .layers import DenseSAKELayer # , SparseSAKELayer
from .utils import ContinuousFilterConvolution, ConcatenationFilter, ContinuousFilterConvolutionWithConcatenation
import math

class DenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        layer: torch.nn.Module=DenseSAKELayer,
        activation: Callable=torch.nn.SiLU(),
        batch_norm: bool=False,
        update_coordinate: Union[List, bool]=False,
        *args, **kwargs,
    ):
        super(DenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
        )
        self.activation = activation
        self.depth = depth
        self.batch_norm = batch_norm
        self.eq_layers = torch.nn.ModuleList()

        if isinstance(update_coordinate, bool):
            update_coordinate = [update_coordinate for _ in range(depth)]

        for idx in range(0, depth):
            self.eq_layers.append(
                layer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    update_coordinate=update_coordinate[idx],
                    *args, **kwargs,
                )
            )

    def forward(self, h, x, mask: Union[None, torch.Tensor]=None):
        h = self.embedding_in(h)
        for idx, eq_layer in enumerate(self.eq_layers):
            h, x = eq_layer(h, x, mask=mask)
        h = self.embedding_out(h)

        return h, x




class VelocityDenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        layer: torch.nn.Module=DenseSAKELayer,
        activation: Callable=torch.nn.SiLU(),
        update_coordinate: Union[List, bool]=False,
        *args, **kwargs,
    ):
        super(VelocityDenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
        )
        self.activation = activation
        self.depth = depth
        self.eq_layers = torch.nn.ModuleList()

        if isinstance(update_coordinate, bool):
            update_coordinate = [update_coordinate for _ in range(depth)]

        for idx in range(0, depth):
            self.eq_layers.append(
                layer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    update_coordinate=update_coordinate[idx],
                    velocity=True,
                    *args, **kwargs,
                )
            )

    def forward(
            self,
            h, x,
            mask: Union[None, torch.Tensor]=None,
            v: Union[None, torch.Tensor]=None,
            h_e_0: Union[None, torch.Tensor]=None,
        ):
        h = self.embedding_in(h)
        for idx, eq_layer in enumerate(self.eq_layers):
            h, x, v = eq_layer(h, x, v, mask=mask, h_e_0=h_e_0)
        h = self.embedding_out(h)
        return h, x


class HighDimensionVelocityDenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        layer: torch.nn.Module=DenseSAKELayer,
        activation: Callable=torch.nn.SiLU(),
        update_coordinate: Union[List, bool]=False,
        n_channel=16,
        *args, **kwargs,
    ):
        super(HighDimensinoVelocityDenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
        )
        self.activation = activation
        self.depth = depth
        self.n_channels = n_channels
        self.eq_layers = torch.nn.ModuleList()
        self.x_mixings = torch.nn.ModuleList()


        if isinstance(update_coordinate, bool):
            update_coordinate = [update_coordinate for _ in range(depth)]

        for idx in range(0, depth):
            self.eq_layers.append(
                layer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    update_coordinate=update_coordinate[idx],
                    velocity=True,
                    *args, **kwargs,
                )
            )

            self.x_mixings.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_features, hidden_features),
                    activation,
                    torch.nn.Linear(hidden_features, n_channels),
                )
            )


    def forward(
            self,
            h, x,
            mask: Union[None, torch.Tensor]=None,
            v: Union[None, torch.Tensor]=None,
            h_e_0: Union[None, torch.Tensor]=None,
        ):


        # (n_batch, n_atoms, d)
        h = self.embedding_in(h)

        x = torch.repeat_interleave(x.unsqueeze(-1), self.n_channels, dim=-1)

        for eq_layer, x_mixing in zip(self.eq_layers, self.x_mixings):

            h, x, v = eq_layer(h, x, v, mask=mask, h_e_0=h_e_0)

            # (..., n_atoms, n_channels)
            lambs = x_mixing(h)

            # (..., n_atoms, n_channel, 3)
            x = x.reshape(
                *x.shape[:-1],
                self.n_channels,
                -1
            )

            x = x * lambs.unsqueeze(-1)

            x = x.flatten(-2, -1)

        h = self.embedding_out(h)
        return h, x



class MultiChannelVelocityDenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        layer: torch.nn.Module=DenseSAKELayer,
        activation: Callable=torch.nn.SiLU(),
        update_coordinate: Union[List, bool]=True,
        n_channels: int=4,
        *args, **kwargs,
    ):
        super(MultiChannelVelocityDenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features*n_channels)
        self.embedding_out = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
        )
        self.activation = activation
        self.depth = depth
        self.eq_layers = torch.nn.ModuleList()
        self.h_mixings = torch.nn.ModuleList()

        self.n_channels = n_channels
        if isinstance(update_coordinate, bool):
            update_coordinate = [update_coordinate for _ in range(depth)]

        for idx in range(0, depth):
            self.eq_layers.append(
                layer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    update_coordinate=update_coordinate[idx],
                    velocity=True,
                    residual=False,
                    *args, **kwargs,
                )
            )

            self.h_mixings.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_channels * hidden_features, hidden_features),
                    activation,
                    torch.nn.Linear(hidden_features, n_channels * hidden_features),
                ),
            )


    def forward(
            self,
            h, x,
            mask: Union[None, torch.Tensor]=None,
            v: Union[None, torch.Tensor]=None,
            h_e_0: Union[None, torch.Tensor]=None,
        ):


        h = self.embedding_in(h)\
            .reshape(*h.shape[:-1], self.hidden_features, self.n_channels)\
            .transpose(-1, -2).transpose(-2, -3)

        x = x.unsqueeze(-3).repeat_interleave(self.n_channels, dim=-3)

        for eq_layer, h_mixing in zip(self.eq_layers, self.h_mixings):
            h, x, v = eq_layer(h, x, v, mask=mask, h_e_0=h_e_0)
            h0 = h

            # -1 hidden, -2 atom, -3 channel, -4 batch
            h = h.reshape(
                *h.shape[:-3],
                h.shape[-2],
                -1,
            )

            h = h_mixing(h)
            h = h.reshape(
                *h.shape[:-2],
                self.n_channels,
                h.shape[-2],
                -1,
            )

            h = h + h0

        h = h.mean(dim=-3)
        x = x.mean(dim=-3)

        h = self.embedding_out(h)
        return h, x



class TandemDenseSAKEModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int=4,
        activation: Callable=torch.nn.SiLU(),
        distance_filter: Callable=ContinuousFilterConvolutionWithConcatenation,
        share_parameters: bool=True,
        *args, **kwargs,
    ):
        super(TandemDenseSAKEModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
        )
        self.activation = activation
        self.depth = depth
        self.share_parameters = share_parameters

        self.eq_layers = torch.nn.ModuleList()
        for idx in range(0, depth):
            self.eq_layers.append(
                DenseSAKELayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    distance_filter=distance_filter,
                    update_coordinate=True,
                    residual=False,
                    *args, **kwargs,
                )
            )

        if share_parameters:
            self.in_layers = self.eq_layers
        else:
            self.in_layers = torch.nn.ModuleList()
            for idx in range(0, depth):
                self.in_layers.append(
                    DenseSAKELayer(
                        in_features=hidden_features,
                        hidden_features=hidden_features,
                        out_features=hidden_features,
                        distance_filter=distance_filter,
                        update_coordinate=False,
                        residual=False,
                        *args, **kwargs,
                    )
                )


    def forward(self, h, x, mask: Union[None, torch.Tensor]=None):
        h = self.embedding_in(h)
        x0 = x
        for eq_layer, in_layer in zip(self.eq_layers, self.in_layers):
            h_eq, x = eq_layer(h, x, mask=mask)
            h_in, _ = in_layer(h, x0, mask=mask, update_coordinate=False)
            h = h_in + h_eq + h

        h = self.embedding_out(h)
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
