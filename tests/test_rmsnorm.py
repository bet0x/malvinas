import torch

from malvinas.norm import RMSNorm


def test_normalizes_to_unit_rms_before_scaling():
    norm = RMSNorm(dim=4, eps=0.0)
    norm.weight.data.fill_(1.0)

    x = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    out = norm(x)

    # RMS of [3,0,0,0] is sqrt((9+0+0+0)/4) = 1.5, so normalized value is 3/1.5 = 2.0
    expected = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)


def test_scales_by_learned_weight():
    norm = RMSNorm(dim=4, eps=0.0)
    norm.weight.data = torch.tensor([2.0, 1.0, 1.0, 1.0])

    x = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    out = norm(x)

    # normalized value is 2.0 (as above), then scaled by weight[0]=2.0 -> 4.0
    expected = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)
