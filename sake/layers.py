import numpy as np
import torch
import dgl
from typing import Callable, Union
import itertools

AGGREGATORS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': lambda *args, **kwargs: torch.max(*args, **kwargs)[0],
    'min': lambda *args, **kwargs: torch.min(*args, **kwargs)[0],
    'var': torch.var,
}

class PNA(torch.nn.Module):
    def __init__(self, aggregators=['sum', 'mean', 'max', 'min', 'var']):
        super(PNA, self).__init__()
        self.aggregators = aggregators
        self.n_aggregators = len(self.aggregators)

    def forward(self, x, dim=1):
        x = torch.cat(
            [
                AGGREGATORS[aggregator](x, dim=dim)
                for aggregator in self.aggregators
            ],
            dim=-1
        )

        return x

class Coloring(torch.nn.Module):
    def __init__(
        self,
        mu=0.0,
        sigma=1.0,
    ):
        super(Coloring, self).__init__()
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sigma", torch.tensor(sigma))

    def forward(self, x):
        return self.sigma * x + self.mu

class ContinuousFilterConvolution(torch.nn.Module):
    def __init__(
            self,
            gamma=10.0,
            mu=torch.arange(0, 30, 1.0),
            ):
        super(ContinuousFilterConvolution, self).__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("mu", torch.tensor(mu))
        self.out_features = len(mu)

    def forward(self, x):
        return torch.exp(
            -(x-self.mu.view(*[1 for _ in range(x.dim()-1)], -1)).pow(2) * self.gamma
        )


class HardCutOff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(HardCutOff, self).__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff))

    def forward(self, x):
        return torch.where(
            torch.gt(x, 0.0) * torch.lt(x, self.cutoff),
            1.0,
            0.0,
        )

class DenseSAKELayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            activation: Callable=torch.nn.SiLU(),
            update_coordinate: bool=True,
            distance_filter: Union[Callable, None]=None,
            n_coefficients: int=64,
            cutoff=None
        ):
        super(DenseSAKELayer, self).__init__()

        self.distance_filter = distance_filter
        self.n_coefficients = n_coefficients
        self.cutoff = cutoff

        self.edge_weight_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_features, n_coefficients),
            torch.nn.Tanh(),
        )

        if self.distance_filter is not None:
            self.distance_filter = self.distance_filter()
            distance_encoding_dimension = self.distance_filter.out_features

        else:
            distance_encoding_dimension = 1

        self.edge_summary_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*in_features+distance_encoding_dimension, hidden_features),
            activation,
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features + hidden_features + n_coefficients, hidden_features),
            # activation,
            # torch.nn.Linear(hidden_features, hidden_features),
        )

        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1)
        )

        self.semantic_attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*in_features, 1, bias=False),
            torch.nn.LeakyReLU(),
        )

        self.update_coordinate = update_coordinate


    def forward(self, h, x):
        # x.shape = (n, 3)
        # h.shape = (n, d)

        # (n, n, 3)
        x_minus_xt = x.unsqueeze(-3) - x.unsqueeze(-2)

        # (n, n, 1)
        x_minus_xt_norm = (x_minus_xt.pow(2).sum(dim=-1, keepdims=True).relu() + 1e-14).pow(0.5)

        # (n, n, 1)
        spatial_att_weights = x_minus_xt_norm.softmax(dim=-2)

        if self.distance_filter is not None:
            x_minus_xt_filtered = self.distance_filter(x_minus_xt_norm)
        else:
            x_minus_xt_filtered = (x_minus_xt_norm + 0.1).pow(-1)

        # (n, n, 2*d)
        h_cat_ht = torch.cat(
            [
                h.unsqueeze(-3).expand(*[-1 for _ in range(h.dim()-2)], h.shape[-2], -1, -1),
                h.unsqueeze(-2).expand(*[-1 for _ in range(h.dim()-2)], -1, h.shape[-2], -1)
            ],
            dim=-1
        )

        # (n, n, 1)
        semantic_att_weights = self.semantic_attention_mlp(h_cat_ht)# .softmax(dim=-2)

        # (n, n, d)
        x_minus_xt_weight = self.edge_weight_mlp(
            h_cat_ht,
        )# .softmax(dim=-2) * 2 - 1.0

        # (n, n, d, 3)
        x_minus_xt_att = x_minus_xt_weight.unsqueeze(-1) * ((x_minus_xt / (x_minus_xt_norm ** 2 + 1e-14)).unsqueeze(-2))

        # (n, d, 3)
        x_minus_xt_att_sum = x_minus_xt_att.sum(dim=-3)

        # (n, d)
        x_minus_xt_att_norm = (x_minus_xt_att_sum.pow(2).sum(-1).relu() + 1e-14).pow(0.5)

        # (n, n, d)
        h_e = self.edge_summary_mlp(
            torch.cat(
                [
                    x_minus_xt_filtered,
                    h_cat_ht
                ],
                dim=-1
            ),
        )

        if self.cutoff is not None:
            cutoff = self.cutoff(x_minus_xt_norm)
            h_e = h_e * cutoff

        if self.update_coordinate is True:
            # (n, 3)
            _x = (x_minus_xt * self.coordinate_mlp(h_e)).sum(dim=-2) + x
        else:
            _x = x

        # (n, d)
        h_e_agg = ((semantic_att_weights * spatial_att_weights) * h_e).sum(dim=-2)


        # (n, d)
        _h = self.node_mlp(
            torch.cat(
                [
                    h,
                    h_e_agg,
                ] + [
                    x_minus_xt_att_norm for _ in range(int(self.n_coefficients>0))
                ],
                dim=-1
            )
        )

        return _h, _x


class GlobalSumSAKELayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        space_dimension: int=3,
        activation: Callable=torch.nn.SiLU(),
    ):
        super(GlobalSumSAKELayer, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1 + 2 * in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )

    def _all_attention_reduce_func(self, node):
        # (n_graph, n_node, 3)
        coordinate = node.mailbox['x']

        # (n_graph, n_node, hidden_dimension)
        feat = node.mailbox['h']

        # (n_graph, n_node, n_node, 1)
        coordinate_attention = (
            coordinate[:, None, :, :] - coordinate[:, :, None, :]
        ).pow(2).sum(dim=-1, keepdims=True)

        # (n_graph, n_node, n_node, hidden_features)
        feat_attention = torch.cat(
            [
                feat[:, None, :, :].expand(-1, coordinate.shape[1], -1, -1),
                feat[:, :, None, :].expand(-1, -1, coordinate.shape[1], -1)
            ],
            dim=-1
        )

        # (n_graph, n_node, n_node, hidden_features)
        all_attention = self.mlp(
            torch.cat(
                [
                    coordinate_attention,
                    feat_attention,
                ],
                dim=-1
            ),
        )

        # (n_graph, hidden_features)
        sum_all_attention = torch.sum(
            all_attention,
            dim=(1, 2)
        )

        return {'sum_all_attention': sum_all_attention}

    def forward(self, graph, feat, coordinate):
        # (n_graph, )
        batch_num_nodes = graph.batch_num_nodes()

        # (n_nodes, )
        graph_idxs = torch.tensor(
            list(itertools.chain(*[[graph_idx for _ in range(num_nodes)] for graph_idx, num_nodes in enumerate(batch_num_nodes)]))
        )

        # constrcut heterograph
        heterograph = dgl.heterograph(
            {
                ('node', 'in', 'graph'): (
                    torch.arange(feat.shape[0]),
                    graph_idxs,
                )
            },
            device=graph.device,
        )

        heterograph.nodes['node'].data['x'] = coordinate
        heterograph.nodes['node'].data['h'] = feat

        heterograph.update_all(
            message_func = lambda edges: {
                'x': edges.src['x'],
                'h': edges.src['h']
            },
            reduce_func = self._all_attention_reduce_func,
        )

        sum_all_attention = heterograph.nodes['graph'].data['sum_all_attention']
        return sum_all_attention

class EGNNLayer(torch.nn.Module):
    """ Layer of E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------
    in_features : int
        Input features.

    out_features : int
        Output features.

    edge_features : int
        Edge features.

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    [2] https://github.com/vgsatorras/egnn

    """
    def __init__(
        self,
        in_features : int,
        hidden_features: int,
        out_features : int,
        edge_features: int=0,
        activation : Callable=torch.nn.SiLU(),
        space_dimension : int=3,
        update_coordinate: bool=True,
        *args, **kwargs,
    ):
        super(EGNNLayer, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.activation = activation
        self.space_dimension = space_dimension
        self.update_coordinate = update_coordinate

        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1)
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                edge_features + in_features * 2 + 1,
                hidden_features
            ),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features)
        )

        self.velocity_nlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1),
        )

    def _edge_model(self, edge):

        return {"h_e":
            self.edge_mlp(
                torch.cat(
                    [edge.data["h_e_0"] for _ in range(int(self.edge_features>0))]
                    + [
                        edge.src["h_v"],
                        edge.dst["h_v"],
                        (edge.src["x"] - edge.dst["x"]).pow(2).sum(
                            dim=-1, keepdims=True
                        ),
                    ],
                    dim=-1
                )
            )
        }

    def _node_model(self, node):
        return {"h_v":
            self.node_mlp(
                torch.cat(
                    [
                        node.data["h_v"],
                        node.data["h_agg"],
                    ],
                    dim=-1,
                )
            )
        }

    def _coordinate_edge_model(self, edge):
        return {
            "x_e": (edge.src["x"] - edge.dst["x"])
            * self.coordinate_mlp(edge.data["h_e"])
        }

    def _coordinate_node_model(self, node):
        return {
            "x": node.data["x"] + node.data["x_agg"],
        }

    def _velocity_and_coordinate_node_model(self, node):
        v = self.velocity_nlp(node.data["h_v"]) * node.data["v"]\
            + node.data["x_agg"]

        return {
            "x": node.data["x"] + v,
            "v": v
        }

    def forward(
            self, graph, feat, coordinate, velocity=None, edge_feat=None
        ):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        velocity : torch.Tensor
            Input velocity.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.
        """
        # get local copy of the graph
        graph = graph.local_var()

        # put features and coordinates into graph
        graph.ndata["h_v"], graph.ndata["x"] = feat, coordinate

        # put edge features into graph
        if edge_feat is not None:
            graph.edata["h_e_0"] = edge_feat

        # apply representation update on edge
        # Eq. 3 in "E(n) Equivariant Graph Neural Networks"
        graph.apply_edges(func=self._edge_model)

        # apply coordinate update on edge
        graph.apply_edges(func=self._coordinate_edge_model)

        # aggregate coordinate update
        graph.update_all(
            dgl.function.copy_e("x_e", "x_msg"),
            dgl.function.sum("x_msg", "x_agg"),
        )

        # apply coordinate update on nodes
        if self.update_coordinate is True:
            if velocity is not None:
                graph.ndata["v"] = velocity
                graph.apply_nodes(func=self._velocity_and_coordinate_node_model)
            else:
                graph.apply_nodes(func=self._coordinate_node_model)

        ## aggregate representation update
        graph.update_all(
            dgl.function.copy_e("h_e", "h_msg"),
            dgl.function.sum("h_msg", "h_agg"),
        )

        # apply representation update on nodes
        graph.apply_nodes(func=self._node_model)

        # pull features
        feat = graph.ndata["h_v"]
        coordinate = graph.ndata["x"]

        return feat, coordinate


class SAKELayer(EGNNLayer):
    """ A SAKE Layer. """
    def __init__(
        self,
        in_features : int,
        hidden_features: int,
        out_features : int,
        edge_features: int=0,
        activation : Callable=torch.nn.SiLU(),
        space_dimension : int=3,
        update_coordinate: bool=True,
        max_in_degree: int=10,
        space_hidden_dimension: int=8,
    ):
        super(SAKELayer, self).__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            activation=activation,
            space_dimension=space_dimension,
            edge_features=edge_features,
            update_coordinate=update_coordinate,
        )

        self.delta_coordinate_model = torch.nn.Sequential(
            torch.nn.Linear(1, space_hidden_dimension),
            activation,
            torch.nn.Linear(space_hidden_dimension, space_hidden_dimension),
            activation,
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(space_hidden_dimension + hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features)
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                space_hidden_dimension + edge_features + in_features * 2 + 1,
                hidden_features
            ),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
        )

        self.node_summary_model = torch.nn.Sequential(
            torch.nn.Linear(5 * space_hidden_dimension, space_hidden_dimension),
            activation,
        )

        self.edge_summary_model = torch.nn.Sequential(
            torch.nn.Linear(5 * space_hidden_dimension, space_hidden_dimension),
            activation,
        )

        self.max_in_degree = max_in_degree
        self.space_hidden_dimension = space_hidden_dimension

    def _copy_x_and_id(self, edges):
        return {'x_msg': edges.src['x'], 'id_msg': edges.edges()[2]}

    def _coordinate_attention(self, nodes):
        # (n_nodes, n_in, space_dimension)
        x_msg = nodes.mailbox['x_msg']

        if x_msg.shape[1] > 1:

            # (n_nodes, n_in, n_in)
            delta_x_msg = (x_msg[:, None, :, :] - x_msg[:, :, None, :]).pow(2).sum(dim=-1)
            # delta_x_msg = delta_x_msg.softmax(dim=-1) * (delta_x_msg.sign().abs())
            #
            delta_x_msg = delta_x_msg / (delta_x_msg.sum() + 1.0) # stablize

            # (n_nodes, n_in, n_in, hidden_dimension)
            h_delta_x = self.delta_coordinate_model(
                delta_x_msg[:, :, :, None]
            )

            # (n_nodes, n_in, hidden_dimension)
            h_e_delta_x = self.edge_summary_model(PNA()(h_delta_x))

            # (n_nodes, hidden_dimension)
            h_v_delta_x = self.node_summary_model(PNA()(h_e_delta_x))

            # padding
            padding = self.max_in_degree - delta_x_msg.shape[1]

            h_e_delta_x = torch.nn.ConstantPad1d(
               (0, padding),
                0.0,
            )(h_e_delta_x.permute(0, 2, 1)).permute(0, 2, 1)

        else:
            # padding
            padding = self.max_in_degree - x_msg.shape[1]

            h_v_delta_x = torch.zeros(
                x_msg.shape[0],
                self.space_hidden_dimension,
                dtype=x_msg.dtype,
                device=x_msg.device,
            )

            h_e_delta_x = torch.zeros(
                x_msg.shape[0],
                self.max_in_degree,
                self.space_hidden_dimension,
                dtype=x_msg.dtype,
                device=x_msg.device,
            )

        # query id
        id_msg = nodes.mailbox['id_msg']

        # pad id
        id_msg = torch.cat([id_msg, -1*torch.ones(x_msg.shape[0], padding, dtype=id_msg.dtype, device=id_msg.device)], dim=-1)


        return {'h_v_delta_x': h_v_delta_x, 'h_e_delta_x': h_e_delta_x, 'edge_id': id_msg}

    def _rearrange_coordinate_attention(self, graph):
        edge_id = graph.ndata['edge_id'].flatten()

        # (n_nodes * max_in_degree, hidden_dimension)
        h_e_delta_x = graph.ndata['h_e_delta_x'].flatten(
            start_dim=0, end_dim=1
        )

        h_e_delta_x = h_e_delta_x[edge_id != -1]
        edge_id = edge_id[edge_id != -1]

        h_e_delta_x_ = torch.empty(
            graph.number_of_edges(),
            self.space_hidden_dimension,
            dtype=h_e_delta_x.dtype,
            device=h_e_delta_x.device,
        )

        h_e_delta_x_[edge_id, :] = h_e_delta_x
        graph.edata['h_e_delta_x'] = h_e_delta_x_
        return graph

    def _node_model(self, node):
        return {"h_v":
            self.node_mlp(
                torch.cat(
                    [
                        node.data["h_v"],
                        node.data["h_agg"],
                        node.data["h_v_delta_x"],
                    ],
                    dim=-1,
                )
            )
        }

    def _edge_model(self, edge):

        return {"h_e":
            self.edge_mlp(
                torch.cat(
                    [edge.data["h_e_0"] for _ in range(int(self.edge_features>0))]
                    + [
                        edge.data["h_e_delta_x"],
                        edge.src["h_v"],
                        edge.dst["h_v"],
                        (edge.src["x"] - edge.dst["x"]).pow(2).sum(
                            dim=-1, keepdims=True
                        ),
                    ],
                    dim=-1
                )
            )
        }

    def forward(
            self, graph, feat, coordinate, velocity=None, edge_feat=None,
        ):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        velocity : torch.Tensor
            Input velocity.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.
        """
        # get local copy of the graph
        graph = graph.local_var()

        # put features and coordinates into graph
        graph.ndata["h_v"], graph.ndata["x"] = feat, coordinate

        # put edge features into graph
        if edge_feat is not None:
            graph.edata["h_e_0"] = edge_feat

        # conduct spatial attention
        graph.update_all(
            message_func=self._copy_x_and_id,
            reduce_func=self._coordinate_attention,
        )

        # rearrange
        graph = self._rearrange_coordinate_attention(graph)

        # apply representation update on edge
        # Eq. 3 in "E(n) Equivariant Graph Neural Networks"
        graph.apply_edges(func=self._edge_model)

        # apply coordinate update on edge
        graph.apply_edges(func=self._coordinate_edge_model)

        # aggregate coordinate update
        graph.update_all(
            dgl.function.copy_e("x_e", "x_msg"),
            dgl.function.sum("x_msg", "x_agg"),
        )

        # apply coordinate update on nodes
        if self.update_coordinate is True:
            if velocity is not None:
                graph.ndata["v"] = velocity
                graph.apply_nodes(func=self._velocity_and_coordinate_node_model)
            else:
                graph.apply_nodes(func=self._coordinate_node_model)

        ## aggregate representation update
        graph.update_all(
            dgl.function.copy_e("h_e", "h_msg"),
            dgl.function.sum("h_msg", "h_agg"),
        )

        # apply representation update on nodes
        graph.apply_nodes(func=self._node_model)

        # pull features
        feat = graph.ndata["h_v"]
        coordinate = graph.ndata["x"]

        return feat, coordinate
