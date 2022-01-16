import pytest
import numpy.testing as npt
from sake.utils import assert_almost_equal_tensor

def test_mp_invariant(_equivariance_test_utils):
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    layer = sake.flow.SAKEFlowLayer(7, 6, 5, residual=False, update_coordinate=True, velocity=True)
    h_original = layer.mp(h0, x0)
    h_translation = layer.mp(h0, translation(x0))
    h_rotation = layer.mp(h0, rotation(x0))
    h_reflection = layer.mp(h0, reflection(x0))

    assert_almost_equal_tensor(h_translation, h_original, decimal=3)
    assert_almost_equal_tensor(h_rotation, h_original, decimal=3)
    assert_almost_equal_tensor(h_reflection, h_original, decimal=3)

def test_layer_forward_backward_same(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    layer = sake.flow.SAKEFlowLayer(7, 7, 5, residual=False, update_coordinate=True, velocity=True)

    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = layer.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(_x0, x0, decimal=3)
    assert_almost_equal_tensor(_v0, v0, decimal=3)
    assert_almost_equal_tensor(log_det_fwd, log_det_bwd)

def test_model_forward_backward_same(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    layer = sake.flow.SAKEFlowModel(7, 7, 5)

    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = layer.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(_x0, x0, decimal=3)
    assert_almost_equal_tensor(_v0, v0, decimal=3)
    assert_almost_equal_tensor(log_det_fwd, log_det_bwd)

def test_layer_jacobian(_equivariance_test_utils):
    import torch
    import sake
    x0 = torch.randn(1, 3)
    v0 = torch.randn_like(x0)
    h0 = torch.randn(1, 7)
    layer = sake.flow.SAKEFlowLayer(7, 7, 5, residual=False, update_coordinate=True, velocity=True)
    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)

    def fn(x_and_v):
        x, v = x_and_v.split(3, dim=-1)
        x1, v1, log_det_fwd = layer.f_forward(h0, x, v)
        return torch.cat([x1, v1], dim=-1)

    x_and_v = torch.cat([x0, v0], dim=-1)
    x_and_v.requires_grad = True

    autograd_jacobian = torch.autograd.functional.jacobian(fn, x_and_v)

    assert_almost_equal_tensor(
        autograd_jacobian.reshape(6, 6).det().abs().log(), log_det_fwd
    )


def test_model_jacobian(_equivariance_test_utils):
    import torch
    import sake
    x0 = torch.randn(1, 3)
    v0 = torch.randn_like(x0)
    h0 = torch.randn(1, 7)
    layer = sake.flow.SAKEFlowModel(7, 7, 5)
    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)

    def fn(x_and_v):
        x, v = x_and_v.split(3, dim=-1)
        x1, v1, log_det_fwd = layer.f_forward(h0, x, v)
        return torch.cat([x1, v1], dim=-1)

    x_and_v = torch.cat([x0, v0], dim=-1)
    x_and_v.requires_grad = True

    autograd_jacobian = torch.autograd.functional.jacobian(fn, x_and_v)

    assert_almost_equal_tensor(
        autograd_jacobian.reshape(6, 6).det().abs().log(), log_det_fwd
    )
