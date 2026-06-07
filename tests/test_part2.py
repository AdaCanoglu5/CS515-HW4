import torch

from part2.channel import power_normalize
from part2.model import K, M, SIGMA2, FeedbackCodeSystem


def test_power_normalize():
    x = torch.randn(1000, 4) * 3.7
    y = power_normalize(x)
    emp = (y**2).sum(dim=1).mean().item()
    assert abs(emp - 1.0) < 0.05, f"power constraint violated: {emp}"


def test_feedback_code_forward_shape_and_finite():
    model = FeedbackCodeSystem()
    m = torch.randint(0, M, (8, K))
    logits = model(m, SIGMA2**0.5)
    assert logits.shape == (8, K, M)
    assert torch.isfinite(logits).all()
