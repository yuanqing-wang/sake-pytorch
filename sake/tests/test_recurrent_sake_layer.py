import pytest
import numpy.testing as npt

def test_simple_forward_t0(_equivariance_test_utils):
    import sake
    h0, x0, _, __, ___ = _equivariance_test_utils
    x0 = x0.unsqueeze(0)
    layer = sake.RecurrentDenseSAKELayer(7, 6, 5, residual=False)
    h, x = layer(h0, x0)
    assert h.shape[0] == 5 == x.shape[1]
    assert x.shape[2] == 3
    assert h.shape[1] == 6


def test_equivariance_t0(_equivariance_test_utils):
    import sake
    from sake.utils import assert_almost_equal_tensor
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    x0 = x0.unsqueeze(0)
    layer = sake.RecurrentDenseSAKELayer(7, 6, 5, residual=False, update_coordinate=True)

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
