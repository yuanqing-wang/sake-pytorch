import torch
import dgl
from typing import Callable
from .layers import EGNNLayer, SAKELayer

class EGNN(torch.nn.Module):
    """ E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth=4,
        edge_features=0,
        activation=torch.nn.SiLU(),
        layer=EGNNLayer,
        max_in_degree=10,
        sum_readout=None,
    ):
        super(EGNN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.activation = activation
        self.depth = depth
        self.max_in_degree = max_in_degree
        self.sum_readout = sum_readout

        for idx in range(0, depth):
            self.add_module(
                "EqLayer_%s" % idx, layer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    activation=activation,
                    edge_features=edge_features,
                    max_in_degree=max_in_degree,
                )
            )

    def forward(self, graph, feat, coordinate, velocity=None, edge_feat=None):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.

        """
        graph = graph.local_var()
        feat = self.embedding_in(feat)
        for idx in range(self.depth):
            feat, coordinate = self._modules["EqLayer_%s" % idx](
                graph, feat, coordinate, velocity, edge_feat,
            )
        feat = self.embedding_out(feat)

        if self.sum_readout is not None:
            graph.ndata['h'] = feat
            feat = dgl.sum_nodes(graph, 'h')
            feat = self.sum_readout(feat)
        return feat, coordinate
