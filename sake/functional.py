import torch

EPSILON = 1e-5
INF = 1e5

def get_x_minus_xt(x):
    return x.unsqueeze(-3) - x.unsqueeze(-2)

def get_x_minus_xt_norm(
    x=None,
    x_minus_xt=None,
    epsilon=EPSILON,
):
    assert x is None or x_minus_xt is None
    assert x is not None or x_minus_xt is not None

    if x_minus_xt is None:
        x_minus_xt = get_x_minus_xt(x)

    x_minus_xt_norm = (
        x_minus_xt.pow(2).sum(dim=-1, keepdim=True).relu()
        + epsilon
    ).pow(0.5)

    return x_minus_xt_norm

def get_h_cat_h(h):
    h_cat_ht = torch.cat(
        [
            h.unsqueeze(-3).repeat_interleave(h.shape[-2], -3),
            h.unsqueeze(-2).repeat_interleave(h.shape[-2], -2),
        ],
        dim=-1
    )

    return h_cat_ht
