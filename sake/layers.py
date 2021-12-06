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
            torch.nn.Linear(2 * in_features + hidden_features + edge_features, n_coefficients),
            torch.nn.Tanh(),
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
            torch.nn.Linear(hidden_features, 1),
            # torch.nn.Tanh(),
        )

        self.semantic_attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*in_features + edge_features, 1, bias=False),
            activation,
        )

        self.update_coordinate = update_coordinate

        self.inf = 1e10
        self.epsilon = 1e-5


class SparseSAKELayer(SAKELayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, g, h, x):
        g.ndata['h'], g.ndata['x'] = h, x

        g.apply_edges(dgl.function.u_sub_v('x', 'x', 'x_minus_xt'))

        g.apply_edges(
            lambda edges: {
                'x_minus_xt_norm':
                (edges.data['x_minus_xt'].pow(2).sum(dim=-1, keepdim=True).relu() + self.epsilon).pow(0.5)
            }
        )

        g.apply_edges(
            lambda edges: {"-x_minus_xt_norm": -edges.data['x_minus_xt_norm']},
        )

        g.apply_edges(
            lambda edges: {"h_cat_ht": torch.cat([edges.src['h'], edges.dst['h']], dim=-1)},
        )

        g.apply_edges(
            lambda edges: {
                "h_e":
                self.distance_filter(
                    edges.data["h_cat_ht"],
                    edges.data["x_minus_xt_norm"]
                )
            }
        )

        g.edata["euclidean_weights"] = dgl.nn.functional.edge_softmax(
            g, g.edata["-x_minus_xt_norm"]
        )

        g.edata["semantic_weights"] = dgl.nn.functional.edge_softmax(
            g, self.semantic_attention_mlp(g.edata["h_cat_ht"]),
        )

        g.apply_edges(
            lambda edges: {
                "total_attention_weights": edges.data["euclidean_weights"] * edges.data["semantic_weights"]
            }
        )

        g.edata["total_attention_weights"] = dgl.nn.functional.edge_softmax(
            g, g.edata["total_attention_weights"],
        )

        g.apply_edges(
            lambda edges: {
                "higher_order_weights":
                self.edge_weight_mlp(
                    torch.cat(
                        [
                            edges.data["h_cat_ht"],
                            edges.data["h_e"]
                        ],
                        dim=-1
                    )
                ),
                "h_e_att": edges.data["total_attention_weights"] * edges.data["h_e"]
            }
        )

        g.apply_edges(
            lambda edges: {
                "x_minus_xt_att":
                edges.data["higher_order_weights"].unsqueeze(-1)\
                * (
                    edges.data["x_minus_xt_norm"].pow(-2)\
                    * edges.data["x_minus_xt"]
                ).unsqueeze(-2)
            }
        )

        g.update_all(
            message_func=dgl.function.copy_e("x_minus_xt_att", "x_minus_xt_att"),
            reduce_func=dgl.function.sum("x_minus_xt_att", "x_minus_xt_att_sum"),
            apply_node_func=lambda nodes: {
                "x_minus_xt_att_norm_embedding": self.post_norm_nlp(
                    (nodes.data["x_minus_xt_att_sum"].pow(2).sum(-1).relu() + self.epsilon).pow(0.5)
                )
            }
        )


        g.update_all(
            message_func=dgl.function.copy_e("h_e_att", "h_e_att"),
            reduce_func=dgl.function.sum("h_e_att", "h_e_agg"),
        )

        g.apply_nodes(
            lambda nodes: {
                "h": self.node_mlp(
                    torch.cat(
                        [
                            nodes.data["h"],
                            nodes.data["h_e_agg"],
                            nodes.data["x_minus_xt_att_norm_embedding"]
                        ],
                        dim=-1,
                    )
                )
            }
        )

        if self.update_coordinate is True:
            g.apply_edges(
                lambda edges: {
                    "x_e": edges.data["x_minus_xt"] * self.coordinate_mlp(
                            edges.data["h_e"],
                    ),
                }
            )

            g.update_all(
                message_func=dgl.function.copy_e("x_e", "x_e"),
                reduce_func=dgl.function.sum("x_e", "x_e"),
                apply_node_func=lambda nodes: {"x": nodes.data["x"] + nodes.data["x_e"]}
            )

        return g.ndata["h"], g.ndata["x"]

class DenseSAKELayer(SAKELayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, h, x, mask=None):
        # x.shape = (n, 3)
        # h.shape = (n, d)

        # (n, n, 3)
        x_minus_xt = x.unsqueeze(-3) - x.unsqueeze(-2)

        # (n, n, 1)
        x_minus_xt_norm = (x_minus_xt.pow(2).sum(dim=-1, keepdim=True).relu() + self.epsilon).pow(0.5)
        _x_minus_xt_norm = x_minus_xt_norm + self.inf * torch.eye(
            x_minus_xt_norm.shape[-2],
            x_minus_xt_norm.shape[-2],
            device=x_minus_xt_norm.device,
        ).unsqueeze(-1)

        if mask is not None:
            _x_minus_xt_norm = _x_minus_xt_norm + (1.0-mask.unsqueeze(-1)) * self.inf

        # (n, n, 1)
        spatial_att_weights = torch.nn.Softmin(dim=-2)(_x_minus_xt_norm)

        h_cat_ht = torch.cat(
            [
                h.unsqueeze(-3).repeat_interleave(h.shape[-2], -3),
                h.unsqueeze(-2).repeat_interleave(h.shape[-2], -2),
            ],
            dim=-1
        )

        x_minus_xt_filtered = self.distance_filter(h_cat_ht, x_minus_xt_norm)

        # (n, n, 1)
        semantic_att_weights = self.semantic_attention_mlp(h_cat_ht)
        semantic_att_weights = semantic_att_weights - self.int * torch.eye(
            semantic_att_weights.shape[-2],
            semantic_att_weights.shape[-2],
            device=semantic_att_weights.device,
        ).unsqueeze(-1)
        
        
        if mask is not None:
            semantic_att_weights = semantic_att_weights + (mask.unsqueeze(-1) - 1.0) * self.inf
        semantic_att_weights = semantic_att_weights.softmax(dim=-2)

        # (n, n, d)
        x_minus_xt_weight = self.edge_weight_mlp(
            torch.cat([h_cat_ht, x_minus_xt_filtered], dim=-1),
        )# .softmax(dim=-2)

        # (n, n, d, 3)
        x_minus_xt_att = x_minus_xt_weight.unsqueeze(-1) * ((x_minus_xt / (x_minus_xt_norm ** 2.0 + self.epsilon)).unsqueeze(-2))


        if mask is not None:
            x_minus_xt_att = x_minus_xt_att * mask.unsqueeze(-1).unsqueeze(-1)

        # (n, d, 3)
        x_minus_xt_att_sum = x_minus_xt_att.sum(dim=-3)

        # (n, d)
        x_minus_xt_att_norm = (x_minus_xt_att_sum.pow(2).sum(-1).relu() + self.epsilon).pow(0.5)
        x_minus_xt_att_norm_embedding = self.post_norm_nlp(x_minus_xt_att_norm)


        h_e = x_minus_xt_filtered

        if self.cutoff is not None:
            cutoff = self.cutoff(x_minus_xt_norm)
            h_e = h_e * cutoff

        if mask is not None:
            h_e = h_e * mask.unsqueeze(-1)

        if self.update_coordinate is True:
            # (n, 3)
            _h_e = self.coordinate_mlp(h_e)
            if mask is not None:
                _h_e = _h_e * mask.unsqueeze(-1)
            _x = (x_minus_xt * _h_e).sum(dim=-2) + x
        else:
            _x = x

        # (n, d)
        total_attention_weights = (semantic_att_weights * spatial_att_weights)
        if mask is not None:
            total_attention_weights = total_attention_weights + (mask.unsqueeze(-1) - 1.0) * self.inf
        total_attention_weights = total_attention_weights.softmax(dim=-2)
        h_e_agg = (total_attention_weights * h_e).sum(dim=-2)

        # _h = self.node_mlp(h_e_agg + x_minus_xt_att_norm_embedding)

        # (n, d)
        _h = self.node_mlp(
              torch.cat(
                  [
                      h,
                      h_e_agg,
                      x_minus_xt_att_norm_embedding,
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
