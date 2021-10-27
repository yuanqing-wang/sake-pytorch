import pytest

def test_rbf_derivatives():
    import torch
    import sake

    x0 = torch.distributions.Normal(
        torch.zeros(20, 5, 3),
        torch.ones(20, 5, 3),
    ).sample()
    x0.requires_grad = True

    x = (x0[:, None, :, :] - x0[:, :, None, :]).pow(2).sum(dim=-1).relu().pow(0.5).unsqueeze(-1)

    y = sake.ContinuousFilterConvolution()(x)

    dy_dx0 = torch.autograd.grad(
        y.sum(),
        x0,
        create_graph=True,
    )[0]

    assert (~torch.isnan(dy_dx0)).all()

def test_layer_derivatives():
    import torch
    import sake
    h0 = torch.distributions.Normal(
        torch.zeros(20, 5, 7),
        torch.ones(20, 5, 7),
    ).sample()
    x0 = torch.distributions.Normal(
        torch.zeros(20, 5, 3),
        torch.ones(20, 5, 3),
    ).sample()

    x0.requires_grad = True
    layer = sake.DenseSAKELayer(in_features=7, hidden_features=8, out_features=9, n_coefficients=0)
    h1, x1 = layer(h0, x0)

    dh1_dx0 = torch.autograd.grad(
        h1.sum(),
        x0,
        create_graph=True,
    )[0]

    assert (~torch.isnan(dh1_dx0)).all()

def test_model_derivatives():
    import torch
    import sake
    h0 = torch.distributions.Normal(
        torch.zeros(20, 5, 7),
        torch.ones(20, 5, 7),
    ).sample()
    x0 = torch.distributions.Normal(
        torch.zeros(20, 5, 3),
        torch.ones(20, 5, 3),
    ).sample()

    x0.requires_grad = True

    model = sake.DenseSAKEModel(
            in_features=7,
            hidden_features=128,
            depth=4,
            out_features=1,
            update_coordinate=False,
            n_coefficients=32,
            distance_filter=sake.ContinuousFilterConvolution,
            activation=torch.nn.SiLU(),
            batch_norm=True,
    )

    h1, x1 = model(h0, x0)

    dh1_dx0 = torch.autograd.grad(
        h1.sum(),
        x0,
        create_graph=True,
    )[0]

    assert (~torch.isnan(dh1_dx0)).all()
