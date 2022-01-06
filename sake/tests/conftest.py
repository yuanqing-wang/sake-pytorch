import pytest
import torch

@pytest.fixture
def _equivariance_test_utils():
    h0 = torch.distributions.Normal(
        torch.zeros(5, 7),
        torch.ones(5, 7),
    ).sample()

    x0 = torch.distributions.Normal(
        torch.zeros(5, 3),
        torch.ones(5, 3),
    ).sample()

    x_translation = torch.distributions.Normal(
        torch.zeros(1, 3),
        torch.ones(1, 3),
    ).sample()
    translation = lambda x: x + x_translation

    import math
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()

    rz = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0,                0,               1],
        ]
    )

    ry = torch.tensor(
        [
            [math.cos(beta),   0,               math.sin(beta)],
            [0,                1,               0],
            [-math.sin(beta),  0,               math.cos(beta)],
        ]
    )

    rx = torch.tensor(
        [
            [1,                0,               0],
            [0,                math.cos(gamma), -math.sin(gamma)],
            [0,                math.sin(gamma), math.cos(gamma)],
        ]
    )

    rotation = lambda x: x @ rz @ ry @ rx

    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
    v = torch.tensor([[alpha, beta, gamma]])
    v /= v.norm()

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p

    return h0, x0, translation, rotation, reflection
