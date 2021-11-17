import pytest
import numpy.testing as npt

def test_import():
    import sake

def test_layer_init():
    import sake
    model = sake.SparseSAKEModel(
        in_features=7, hidden_features=8, out_features=9,
    )

# def test_layer_simple_graph_zeros():
#     import torch
#     import dgl
#     import egnn
#     g = dgl.rand_graph(5, 8)
#     h = torch.zeros(5, 7)
#     x = torch.zeros(5, 3)
#     layer = egnn.EGNNLayer(in_features=7, hidden_features=8, out_features=9)
#     h, x = layer(g, h, x)
#     assert (x == 0).all()

def test_layer_simple_graph_equivariant():
    import torch
    import dgl
    import sake
    g = dgl.rand_graph(5, 8)
    h0 = torch.distributions.Normal(
        torch.zeros(5, 7),
        torch.ones(5, 7),
    ).sample()
    x0 = torch.distributions.Normal(
        torch.zeros(5, 3),
        torch.ones(5, 3),
    ).sample()
    net = sake.SparseSAKEModel(in_features=7, hidden_features=8, out_features=9)

    # original
    h_original, x_original = net(g, h0, x0)

    # ~~~~~~~~~~~
    # translation
    # ~~~~~~~~~~~
    translation = torch.distributions.Normal(
        torch.zeros(1, 3),
        torch.ones(1, 3),
    ).sample()

    h_translation, x_translation = net(
        g,
        h0,
        x0 + translation
    )

    npt.assert_almost_equal(h_translation.detach().numpy(), h_original.detach().numpy(), decimal=2)
    npt.assert_almost_equal(x_translation.detach().numpy(), (x_original + translation).detach().numpy(), decimal=2)


    # ~~~~~~~~
    # rotation
    # ~~~~~~~~
    import math
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()

    rz = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0,                0,               1],
        ]
    )

    ry = torch.tensor(
        [
            [math.cos(beta),   0,               math.sin(beta)],
            [0,                1,               0],
            [-math.sin(beta),  0,               math.cos(beta)],
        ]
    )

    rx = torch.tensor(
        [
            [1,                0,               0],
            [0,                math.cos(gamma), -math.sin(gamma)],
            [0,                math.sin(gamma), math.cos(gamma)],
        ]
    )

    h_rotation, x_rotation = net(
        g,
        h0,
        x0 @ rz @ ry @ rx,
    )

    npt.assert_almost_equal(h_rotation.detach().numpy(), h_original.detach().numpy(), decimal=2)
    npt.assert_almost_equal(x_rotation.detach().numpy(), (x_original @ rz @ ry @ rx).detach().numpy(), decimal=2)

    # ~~~~~~~~~~
    # reflection
    # ~~~~~~~~~~
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    v = torch.tensor([[alpha, beta, gamma]])
    v /= v.norm()

    p = torch.eye(3) - 2 * v.T @ v

    h_reflection, x_reflection = net(
        g,
        h0,
        x0 @ p,
    )

    npt.assert_almost_equal(h_reflection.detach().numpy(), h_original.detach().numpy(), decimal=2)
    npt.assert_almost_equal(x_reflection.detach().numpy(), (x_original @ p).detach().numpy(), decimal=2)
