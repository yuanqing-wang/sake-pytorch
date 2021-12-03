import pytest
import numpy.testing as npt

def test_padding_equivalence():
    import torch
    import sake
    x = torch.randn(8, 3)
    h = torch.randn(8, 5)
    layer = sake.DenseSAKELayer(5, 6, 7)
    _h_original, _x_original = layer(h, x)

    x_padded = torch.cat(
        [
            x,
            torch.randn(10, 3),
        ],
        dim=0,
    )

    h_padded = torch.cat(
        [
            h,
            torch.randn(10, 5),
        ],
        dim=0
    )

    mask = torch.cat(
        [
            torch.ones(8),
            torch.zeros(10),
        ],
        dim=0
    )

    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    _h_masked, _x_masked = layer(h_padded, x_padded, mask=mask)

    npt.assert_almost_equal(
        _h_masked[:8].detach().numpy(),
        _h_original.detach().numpy(),
    )

    npt.assert_almost_equal(
        _x_masked[:8].detach().numpy(),
        _x_original.detach().numpy(),
    )
