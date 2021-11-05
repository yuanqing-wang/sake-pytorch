import pytest


def test_bootstrap_zero_one():
    import torch
    import sake
    x = torch.ones(5)
    y = torch.zeros(5)
    original, high, low = sake.bootstrap(torch.nn.MSELoss())(x, y)
    assert original == 1.0 == high == low
