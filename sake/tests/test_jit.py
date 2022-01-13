import pytest
import numpy.testing as npt
import torch

def test_simple_layer_forward(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False)
    layer = torch.jit.script(layer)
    h, x = layer(h0, x0)


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="no cuda")
def test_simple_layer_forward_cuda(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    h0 = h0.cuda()
    x0 = x0.cuda()
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False)
    layer = layer.cuda()
    layer = torch.jit.script(layer)
    h, x = layer(h0, x0)

def test_simple_model_forward(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    model = sake.DenseSAKEModel(7, 6, 5, residual=False)
    model = torch.jit.script(model)
    h, x = model(h0, x0)

@pytest.mark.skipif(torch.cuda.is_available() is False, reason="no cuda")
def test_simple_model_forward_cuda(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    h0 = h0.cuda()
    x0 = x0.cuda()
    model = sake.DenseSAKEModel(7, 6, 5, residual=False)
    model = model.cuda()
    model = torch.jit.script(model)
    h, x = model(h0, x0)

def test_tandem_model_forward(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    model = sake.TandemDenseSAKEModel(7, 7, 7)
    model = torch.jit.script(model)
    h, x = model(h0, x0)

def test_grad(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    model = sake.DenseSAKEModel(7, 6, 5, residual=False)
    model = torch.jit.script(model)
    h, x = model(h0, x0)
    h.sum().backward()

def test_double_grad(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    x0.requires_grad = True
    model = sake.DenseSAKEModel(7, 6, 5, residual=False)
    model = torch.jit.script(model)
    h, x = model(h0, x0)
    dh_dx0 = torch.autograd.grad(
        h.sum(),
        x0,
        create_graph=True,
    )[0]
    dh_dx0.sum().backward()
