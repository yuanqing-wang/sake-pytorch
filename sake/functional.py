import torch

EPSILON = 1e-5
INF = 1e5

def get_x_minus_xt(x):
    return x.unsqueeze(-3) - x.unsqueeze(-2)

def get_x_minus_xt_norm(
    x_minus_xt,
    epsilon: float=EPSILON,
):
    x_minus_xt_norm = (
        x_minus_xt.pow(2).sum(dim=-1, keepdim=True).relu()
        + epsilon
    ).pow(0.5)

    return x_minus_xt_norm

def get_h_cat_h(h):
    n_nodes = int(h.shape[-2])
    h_cat_ht = torch.cat(
        [
            h.unsqueeze(-3).repeat_interleave(n_nodes, -3),
            h.unsqueeze(-2).repeat_interleave(n_nodes, -2),
        ],
        dim=-1
    )

    return h_cat_ht
