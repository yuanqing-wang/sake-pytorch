import pytest
import numpy.testing as npt

def test_simple_forward(_equivariance_test_utils):
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False)
    h, x = layer(h0, x0)
    assert h.shape[0] == 5 == x.shape[0]
    assert x.shape[1] == 3
    assert h.shape[1] == 6

def test_equivariance(_equivariance_test_utils):
    import sake
    from sake.utils import assert_almost_equal_tensor
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False, update_coordinate=True)

    h_original, x_original = layer(h0, x0)
    h_translation, x_translation = layer(h0, translation(x0))
    h_rotation, x_rotation = layer(h0, rotation(x0))
    h_reflection, x_reflection = layer(h0, reflection(x0))

    assert_almost_equal_tensor(h_translation, h_original, decimal=3)
    assert_almost_equal_tensor(h_rotation, h_original, decimal=3)
    assert_almost_equal_tensor(h_reflection, h_original, decimal=3)

    assert_almost_equal_tensor(x_translation, translation(x_original), decimal=3)
    assert_almost_equal_tensor(x_rotation, rotation(x_original), decimal=3)
    assert_almost_equal_tensor(x_reflection, reflection(x_original), decimal=3)

def test_equivariance_with_velocity(_equivariance_test_utils):
    import torch
    import sake
    from sake.utils import assert_almost_equal_tensor
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    layer = sake.DenseSAKELayer(7, 6, 5, residual=False, update_coordinate=True, velocity=True)

    h_original, x_original, v_original = layer(h0, x0, v0)
    h_translation, x_translation, v_translation = layer(h0, translation(x0), v0)
    h_rotation, x_rotation, v_rotation = layer(h0, rotation(x0), rotation(v0))
    h_reflection, x_reflection, v_reflection = layer(h0, reflection(x0), reflection(v0))

    assert_almost_equal_tensor(h_translation, h_original, decimal=3)
    assert_almost_equal_tensor(h_rotation, h_original, decimal=3)
    assert_almost_equal_tensor(h_reflection, h_original, decimal=3)

    assert_almost_equal_tensor(x_translation, translation(x_original), decimal=3)
    assert_almost_equal_tensor(x_rotation, rotation(x_original), decimal=3)
    assert_almost_equal_tensor(x_reflection, reflection(x_original), decimal=3)
