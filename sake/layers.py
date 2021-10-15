import torch
import dgl
from typing import Callable

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

    def forward(self, x, dim=1):
        x = torch.cat(
            [
                AGGREGATORS[aggregator](x, dim=dim)
                for aggregator in self.aggregators
            ],
            dim=-1
        )

        return x

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
