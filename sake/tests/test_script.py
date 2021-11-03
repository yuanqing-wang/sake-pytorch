import pytest

def test_simple_script_forward():
    import sake
    import torch
    x = torch.randn(8, 3)
    h = torch.randn(8, 5)
    layer = sake.DenseSAKELayer(5, 6, 7)
    layer = torch.jit.script(layer)
    _x, _h = layer(h, x)
