import numpy as np
import torch
import dgl
from typing import Callable, Union
import itertools
from .utils import PNA, Coloring, RBF, HardCutOff, ContinuousFilterConvolution

class SAKELayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            edge_features: int=0,
            activation: Callable=torch.nn.SiLU(),
            update_coordinate: bool=True,
            distance_filter: Callable=ContinuousFilterConvolution,
            n_coefficients: int=64,
            cutoff=None,
        ):
        super().__init__()

        self.distance_filter = distance_filter
        self.n_coefficients = n_coefficients
        self.cutoff = cutoff

        self.edge_weight_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_features + hidden_features + edge_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, n_coefficients),
        )

        self.distance_filter = self.distance_filter(
            2 * in_features, hidden_features,
        )

        self.post_norm_nlp = torch.nn.Sequential(
            torch.nn.Linear(n_coefficients, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_features + in_features + edge_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            # torch.nn.Linear(hidden_features, hidden_features),
            # activation,
            torch.nn.Linear(hidden_features, 1),
        )

        self.semantic_attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*in_features + edge_features, 1, bias=False),
            activation,
        )

        self.update_coordinate = update_coordinate

        self.inf = 1e10
        self.epsilon = 1e-5
