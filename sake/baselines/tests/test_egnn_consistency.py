import pytest
from typing import Callable
import numpy.testing as npt

import torch
import random
random.seed(2666)

DOWNLOAD_EGNN_CMD = """
rm -rf egnn
git clone https://github.com/vgsatorras/egnn.git
mv egnn/qm9 qm9
mv egnn/models models
rm -rf egnn
"""

def download_official_egnn():
    import os
    os.system(DOWNLOAD_EGNN_CMD)

def remove_official_egnn():
    import os
    os.system("rm -rf qm9")
    os.system("rm -rf models")

@pytest.fixture
def official_egnn_implementation():
    download_official_egnn()
    from qm9.models import EGNN
    model = EGNN(
        in_node_nf=15,
        in_edge_nf=0,
        hidden_nf=128,
        n_layers=7,
        coords_weight=1.0,
        attention=1,
        node_attr=0,
    )

    return model

@pytest.fixture
def official_egnn_dataset():
    from qm9 import dataset
    dataloaders, charge_scale = dataset.retrieve_dataloaders(96, 0)
    return dataloaders, charge_scale

@pytest.fixture
def our_implementation():
    class Readout(torch.nn.Module):
        def __init__(
                self,
                in_features:int=128,
                hidden_features:int=128,
                out_features:int=1,
                activation: Callable=torch.nn.SiLU(),
            ):
            super().__init__()
            self.before_sum = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, hidden_features),
            )
            self.after_sum = torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features),
                activation,
                torch.nn.Linear(hidden_features, out_features),
            )

        def forward(self, h, mask=None):
            h = self.before_sum(h)
            h = h * mask # torch.sign(mask.sum(dim=-1, keepdim=True))
            h = h.sum(dim=-2)
            h = self.after_sum(h)
            return h

    import sake
    from sake.baselines.egnn import EGNNLayer

    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.dense_sake = sake.DenseSAKEModel(
                in_features=15,
                hidden_features=128,
                out_features=128,
                depth=7,
                update_coordinate=False,
                activation=torch.nn.SiLU(),
                layer=EGNNLayer,
                attention=True,
            )

            self.dense_sake.embedding_out = Identity()
            self.readout = Readout(in_features=128, hidden_features=128, out_features=1)

        def forward(self, h, x, edge_mask, node_mask):
            h, _ = self.dense_sake(h, x, mask=edge_mask)
            h = self.readout(h, mask=node_mask)
            return h

    model = Model()
    return model

def test_infrastructure(
        official_egnn_implementation,
        official_egnn_dataset,
        our_implementation,
    ):
    return True

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_number_of_parameters_same(official_egnn_implementation, our_implementation):
    assert get_number_of_parameters(
        official_egnn_implementation,
    ) == get_number_of_parameters(
        our_implementation,
    )

@pytest.fixture
def our_implementation_fixed(official_egnn_implementation, our_implementation):
    for official, our in zip(official_egnn_implementation.named_parameters(), our_implementation.named_parameters()):
        official_name, official_parameter = official
        our_name, our_parameter = our

        with torch.no_grad():
            our_parameter.copy_(official_parameter)

    return our_implementation

def test_identical_forward(
        official_egnn_implementation,
        our_implementation_fixed,
        official_egnn_dataset,
    ):

    from qm9 import utils as qm9_utils
    dataloaders, charge_scale = official_egnn_dataset
    data = next(iter(dataloaders['train']))

    device = "cpu"
    dtype = torch.float32

    batch_size, n_nodes, _ = data['positions'].size()
    atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
    atom_mask = data['atom_mask'].view(batch_size, n_nodes, -1).to(device, dtype)
    edge_mask = data['edge_mask'].to(device, dtype).view(batch_size, n_nodes, n_nodes)
    one_hot = data['one_hot'].to(device, dtype)
    charges = data['charges'].to(device, dtype)
    nodes = qm9_utils.preprocess_input(one_hot, charges, 2, charge_scale, device)

    _atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
    _atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
    _edge_mask = data['edge_mask'].to(device, dtype)
    _nodes = nodes.view(batch_size * n_nodes, -1)
    _edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)

    official_layer = official_egnn_implementation.gcl_0
    our_layer = our_implementation_fixed.dense_sake.eq_layers[0]


    # test edge model
    row, col = _edges
    _h = official_egnn_implementation.embedding(_nodes)
    radial, _ = official_layer.coord2radial(_edges, _atom_positions)
    _h_e = official_layer.edge_model(_h[col], _h[row], radial, None)

    from sake.functional import get_h_cat_h, get_x_minus_xt_norm
    h = our_implementation_fixed.dense_sake.embedding_in(nodes)

    h_cat_ht = get_h_cat_h(h)
    x_minus_xt_norm = get_x_minus_xt_norm(atom_positions, epsilon=0.0)
    h_e = our_layer.edge_model(h_cat_ht, radial.reshape(x_minus_xt_norm.shape))

    npt.assert_almost_equal(
        _h.detach().flatten().numpy(),
        h.detach().flatten().numpy(),
    )

    npt.assert_almost_equal(
        radial.detach().flatten().numpy(),
        x_minus_xt_norm.pow(2).detach().flatten().numpy(),
        decimal=2,
    )

    npt.assert_almost_equal(
        _h_e.detach().flatten().numpy(),
        h_e.detach().flatten().numpy(),
        decimal=2,
    )

    # test node model
    _h_v, _ = official_layer.node_model(_h, _edges, _h_e * _edge_mask, None)
    # def unsorted_segment_sum(data, segment_ids, num_segments):
    #     """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    #     result_shape = (num_segments, data.size(1))
    #     result = data.new_full(result_shape, 0)  # Init empty result tensor.
    #     segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    #     result.scatter_add_(0, segment_ids, data)
    #     return result
    #
    # agg = unsorted_segment_sum(_h_e, row, num_segments=_h.size(0)).view(batch_size, n_nodes, -1)

    h_e = our_layer.aggregate(_h_e.reshape(h_e.shape), edge_mask)
    h_v = our_layer.node_model(h, h_e)

    npt.assert_almost_equal(
        _h_v.detach().flatten().numpy(),
        h_v.detach().flatten().numpy(),
        decimal=2,
    )

    # test overal model
    our_y_hat = our_implementation_fixed(nodes, atom_positions, edge_mask, atom_mask)
    official_y_hat = official_egnn_implementation(
        h0=_nodes,
        x=_atom_positions,
        edges=_edges,
        edge_attr=None,
        node_mask=_atom_mask,
        edge_mask=_edge_mask,
        n_nodes=n_nodes,
    )

    npt.assert_almost_equal(
        our_y_hat.detach().flatten().numpy(),
        official_y_hat.detach().flatten().numpy(),
        decimal=2,
    )

def test_removal():
    remove_official_egnn()
