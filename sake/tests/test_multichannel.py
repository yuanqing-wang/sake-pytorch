import pytest

def test_multichannel_model_equivalent(_equivariance_test_utils):
    import torch
    import sake
    from sake.utils import assert_almost_equal_tensor

    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    model = sake.MultiChannelVelocityDenseSAKEModel(7, 7, 7)
    h, x = model(h0, x0)

    h_original, x_original = model(h0, x0)
    h_translation, x_translation = model(h0, translation(x0))
    h_rotation, x_rotation = model(h0, rotation(x0))
    h_reflection, x_reflection = model(h0, reflection(x0))

    assert_almost_equal_tensor(h_translation, h_original, decimal=3)
    assert_almost_equal_tensor(h_rotation, h_original, decimal=3)
    assert_almost_equal_tensor(h_reflection, h_original, decimal=3)

    assert_almost_equal_tensor(x_translation, translation(x_original), decimal=3)
    assert_almost_equal_tensor(x_rotation, rotation(x_original), decimal=3)
    assert_almost_equal_tensor(x_reflection, reflection(x_original), decimal=3)
