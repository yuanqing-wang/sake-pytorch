import pytest
import numpy.testing as npt

def test_simple_forward():
    import sake
    import torch
    x = torch.randn(8, 3)
    h = torch.randn(8, 5)
    layer = sake.DenseSAKELayer(5, 6, 7, distance_filter=sake.ContinuousFilterConvolution)
    _x, _h = layer(h, x)

def test_layer_simple_graph_equivariant():
    import torch
    import dgl
    import sake
    h0 = torch.distributions.Normal(
        torch.zeros(5, 7),
        torch.ones(5, 7),
    ).sample()
    x0 = torch.distributions.Normal(
        torch.zeros(5, 3),
        torch.ones(5, 3),
    ).sample()
    layer = sake.DenseSAKELayer(in_features=7, hidden_features=8, out_features=9, distance_filter=sake.ContinuousFilterConvolution)

    # original
    h_original, x_original = layer(h0, x0)

    # ~~~~~~~~~~~
    # translation
    # ~~~~~~~~~~~
    translation = torch.distributions.Normal(
        torch.zeros(1, 3),
        torch.ones(1, 3),
    ).sample()

    h_translation, x_translation = layer(
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

    h_rotation, x_rotation = layer(
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

    h_reflection, x_reflection = layer(
        h0,
        x0 @ p,
    )

    npt.assert_almost_equal(h_reflection.detach().numpy(), h_original.detach().numpy(), decimal=2)
    npt.assert_almost_equal(x_reflection.detach().numpy(), (x_original @ p).detach().numpy(), decimal=2)



def test_layer_simple_graph_equivariant_batch():
    import torch
    import dgl
    import sake
    h0 = torch.distributions.Normal(
        torch.zeros(20, 5, 7),
        torch.ones(20, 5, 7),
    ).sample()
    x0 = torch.distributions.Normal(
        torch.zeros(20, 5, 3),
        torch.ones(20, 5, 3),
    ).sample()
    layer = sake.DenseSAKELayer(in_features=7, hidden_features=8, out_features=9, distance_filter=sake.ContinuousFilterConvolution)

    # original
    h_original, x_original = layer(h0, x0)

    # ~~~~~~~~~~~
    # translation
    # ~~~~~~~~~~~
    translation = torch.distributions.Normal(
        torch.zeros(1, 3),
        torch.ones(1, 3),
    ).sample()

    h_translation, x_translation = layer(
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

    h_rotation, x_rotation = layer(
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

    h_reflection, x_reflection = layer(
        h0,
        x0 @ p,
    )

    npt.assert_almost_equal(h_reflection.detach().numpy(), h_original.detach().numpy(), decimal=2)
    npt.assert_almost_equal(x_reflection.detach().numpy(), (x_original @ p).detach().numpy(), decimal=2)
