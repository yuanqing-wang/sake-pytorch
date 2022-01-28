import pytest
import numpy.testing as npt
from sake.utils import assert_almost_equal_tensor

def test_mp_eqvariant(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    layer = sake.flow.SAKEFlowLayer(7, 7)
    h_original, s_original = layer.mp(h0, x0)
    h_rotation, s_rotation = layer.mp(h0, rotation(x0))
    h_reflection, s_reflection = layer.mp(h0, reflection(x0))

    # assert_almost_equal_tensor(h_translation, h_original, decimal=3)
    assert_almost_equal_tensor(h_rotation, h_original, decimal=3)
    assert_almost_equal_tensor(h_reflection, h_original, decimal=3)

    assert_almost_equal_tensor(s_rotation, rotation(s_original), decimal=3)
    assert_almost_equal_tensor(s_reflection, reflection(s_original), decimal=3)

def test_layer_center(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    layer = sake.flow.SAKEFlowLayer(7, 7)

    x0 = x0 - x0.mean(dim=-2, keepdim=True)
    v0 = v0 - v0.mean(dim=-2, keepdim=True)

    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = layer.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(x1.mean(dim=-2), torch.zeros_like(x1.mean(dim=-2)), decimal=3)
    assert_almost_equal_tensor(v1.mean(dim=-2), torch.zeros_like(v1.mean(dim=-2)), decimal=3)

def test_layer_forward_backward_same(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    layer = sake.flow.SAKEFlowLayer(7, 7)

    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = layer.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(_x0, x0, decimal=3)
    assert_almost_equal_tensor(_v0, v0, decimal=3)
    assert_almost_equal_tensor(log_det_fwd, log_det_bwd, decimal=3)

def test_model_forward_backward_same(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    model = sake.flow.SAKEFlowModel(7, 7, depth=4, log_gamma=2.0)

    v0 = v0 - v0.mean(dim=-2, keepdim=True)
    x0 = x0 - x0.mean(dim=-2, keepdim=True)
    x1, v1, log_det_fwd = model.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = model.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(_x0, x0, decimal=3)
    assert_almost_equal_tensor(_v0, v0, decimal=3)
    assert_almost_equal_tensor(log_det_fwd, log_det_bwd, decimal=3)

def test_model_center(_equivariance_test_utils):
    import torch
    import sake
    h0, x0, translation, rotation, reflection = _equivariance_test_utils
    v0 = torch.randn_like(x0)
    model = sake.flow.SAKEFlowModel(7, 7, depth=4, log_gamma=2.0)

    x0 = x0 - x0.mean(dim=-2, keepdim=True)
    v0 = v0 - v0.mean(dim=-2, keepdim=True)

    x1, v1, log_det_fwd = model.f_forward(h0, x0, v0)
    _x0, _v0, log_det_bwd = model.f_backward(h0, x1, v1)

    assert_almost_equal_tensor(x1.mean(dim=-2), torch.zeros_like(x1.mean(dim=-2)), decimal=3)
    assert_almost_equal_tensor(v1.mean(dim=-2), torch.zeros_like(v1.mean(dim=-2)), decimal=3)

# def test_jit_layer(_equivariance_test_utils):
#     import torch
#     import sake
#     layer = sake.flow.SAKEFlowLayer(7, 7, 5)
#     layer.forward = layer.f_forward
#     layer = torch.jit.script(layer)
#
#     h0, x0, translation, rotation, reflection = _equivariance_test_utils
#     v0 = torch.randn_like(x0)
#     x1, v1, log_det_fwd = layer(h0, x0, v0)
#
# def test_jit_model(_equivariance_test_utils):
#     import torch
#     import sake
#     model = sake.flow.SAKEFlowModel(7, 7, depth=1)
#     model.forward = model.f_forward
#     model = torch.jit.script(model)
#
#     h0, x0, translation, rotation, reflection = _equivariance_test_utils
#     v0 = torch.randn_like(x0)
#     x1, v1, log_det_fwd = model.f_forward(h0, x0, v0)


def test_layer_jacobian(_equivariance_test_utils):
    import torch
    import sake
    x0 = torch.randn(1, 3)
    v0 = torch.randn_like(x0)
    h0 = torch.randn(1, 7)
    layer = sake.flow.SAKEFlowLayer(7, 7, 5)
    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)

    def fn(x_and_v):
        x, v = x_and_v.split(3, dim=-1)
        x1, v1, log_det_fwd = layer.f_forward(h0, x, v)
        return torch.cat([x1, v1], dim=-1)

    x_and_v = torch.cat([x0, v0], dim=-1)
    x_and_v.requires_grad = True

    autograd_jacobian = torch.autograd.functional.jacobian(fn, x_and_v)

    assert_almost_equal_tensor(
        autograd_jacobian.reshape(6, 6).det().abs().log(), log_det_fwd,
        decimal=3,
    )


def test_model_jacobian(_equivariance_test_utils):
    import torch
    import sake
    x0 = torch.randn(1, 3)
    v0 = torch.randn_like(x0)
    h0 = torch.randn(1, 7)
    x0 = x0 - x0.mean(dim=-2, keepdim=True)
    v0 = v0 - x0.mean(dim=-2, keepdim=True)
    layer = sake.flow.SAKEFlowModel(7, 7, log_gamma=2.0)
    x1, v1, log_det_fwd = layer.f_forward(h0, x0, v0)

    def fn(x_and_v):
        x, v = x_and_v.split(3, dim=-1)
        x1, v1, log_det_fwd = layer.f_forward(h0, x, v)
        return torch.cat([x1, v1], dim=-1)

    x_and_v = torch.cat([x0, v0], dim=-1)
    x_and_v.requires_grad = True

    autograd_jacobian = torch.autograd.functional.jacobian(fn, x_and_v)
    assert_almost_equal_tensor(
        autograd_jacobian.reshape(6, 6).det().abs().log(), log_det_fwd,
        decimal=3,
    )
