import pytest
import numpy.testing as npt
import torch

def test_simple_forward(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False)
    layer = torch.jit.script(layer)
    h, x = layer(h0, x0)


@pytest.skipif(torch.cuda.is_available() is False, reason="no cuda")
def test_simple_forward(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    h0 = h0.cuda()
    x0 = x0.cuda()
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False)
    layer = layer.cuda()
    layer = torch.jit.script(layer)
    h, x = layer(h0, x0)
