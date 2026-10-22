"""Scientific checks of causal access, probability semantics, and gradients."""

import pytest
import torch
from torch.nn import functional as F

from spectrum.research.losses import classification_loss, joint_probabilities
from spectrum.research.models import build_model


@pytest.fixture(autouse=True)
def deterministic_small_cpu_run():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("architecture,field", [("mlp", 1), ("cnn", 5), ("tcn", 13), ("lstm", None)])
def test_architecture_switch_and_prefix_causality(architecture, field):
    model = build_model(3, dict(architecture=architecture, hidden=8, layers=2, kernel_size=3))
    assert model.receptive_field == field
    x = torch.randn(4, 20, 3)
    model.eval()
    full = model.sequence_logits(x)
    torch.testing.assert_close(model(x[:, :9]), full[:, 8], atol=1e-6, rtol=1e-5)
    disturbed = x.clone()
    disturbed[:, 9:] = 100 * torch.randn_like(disturbed[:, 9:])
    torch.testing.assert_close(model.sequence_logits(disturbed)[:, :9], full[:, :9])
    # Other examples do not change this example's state or normalization.
    torch.testing.assert_close(model(x[:1]), model(x)[:1], atol=1e-6, rtol=1e-5)
    logits = model(x)
    loss = classification_loss(logits, torch.tensor([0, 1, 2, 3]), {})
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("architecture", ["mlp", "cnn", "tcn"])
def test_claimed_receptive_field_excludes_older_samples(architecture):
    model = build_model(2, dict(architecture=architecture, hidden=5, layers=2, kernel_size=3)).eval()
    x = torch.randn(2, 40, 2)
    changed = x.clone()
    changed[:, :-model.receptive_field] += 100
    torch.testing.assert_close(model(x), model(changed))


def test_unknown_model_is_not_silently_replaced():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model(4, {"architecture": "mamba2"})
    with pytest.raises(ValueError, match="positive integer"):
        build_model(4, {"layers": 0})


def test_joint_nll_and_unknown_root_mask_agree():
    z = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, -1])
    p = joint_probabilities(z)
    torch.testing.assert_close(p.sum(1), torch.ones(5, dtype=z.dtype))
    loss = classification_loss(z, y, {})
    expected = -p[torch.arange(4), y[:4]].log().mean()
    torch.testing.assert_close(loss, expected)
    per_root = classification_loss(z, y, {}, reduction="none")
    torch.testing.assert_close(per_root.sum() / 4, loss)
    assert per_root[-1] == 0
    loss.backward()
    assert (z.grad[-1] == 0).all()
    assert (z.grad[0, 1:] == 0).all()
    assert (z.grad[1:4, 1:].abs().sum(1) > 0).all()


def test_focal_gamma_zero_and_brier_zero_recover_bce_gradients():
    y = torch.tensor([0, 1, 2, 3, -1])
    original = torch.randn(5, 4, dtype=torch.float64)
    outcomes = []
    for config in [{}, {"event_loss": "focal", "gamma": 0},
                   {"event_loss": "bce_brier", "brier_lambda": 0},
                   {"event_loss": "proper_confidence", "epsilon": 1}]:
        z = original.clone().requires_grad_()
        loss = classification_loss(z, y, config)
        outcomes.append((loss.detach(), torch.autograd.grad(loss, z)[0]))
    for loss, grad in outcomes[1:]:
        torch.testing.assert_close(loss, outcomes[0][0])
        torch.testing.assert_close(grad, outcomes[0][1])


@pytest.mark.parametrize("event_loss", ["bce", "focal", "bce_brier", "proper_confidence"])
def test_losses_have_finite_gradients_and_pass_finite_differences(event_loss):
    y = torch.tensor([0, 1, 2, 3, -1])
    z = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    config = dict(event_loss=event_loss, gamma=0.5, brier_lambda=0.7, epsilon=1 / 3)
    assert torch.autograd.gradcheck(lambda a: classification_loss(a, y, config), (z,))
    extreme = torch.tensor([[1000., -1000., 1000., 0.], [-1000., 1000., -1000., 0.],
                            [1000., 0., 1000., 0.], [-1000., 0., 0., 0.]],
                           dtype=torch.float64, requires_grad=True)
    loss = classification_loss(extreme, torch.tensor([0, 2, 2, 0]), config)
    grad, = torch.autograd.grad(loss, extreme)
    assert torch.isfinite(loss) and torch.isfinite(grad).all()


def test_confidence_gradient_matches_proper_formula():
    z = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3])
    epsilon = 1 / 3
    loss = classification_loss(z, y, dict(event_loss="proper_confidence", epsilon=epsilon))
    gradient, = torch.autograd.grad(loss, z)
    p = z[:, 0].sigmoid()
    expected = (p - (y > 0).to(p.dtype)) * (epsilon + (1 - epsilon) * (2 * p - 1).square()) / 4
    torch.testing.assert_close(gradient[:, 0], expected)
    # Under Bernoulli frequency q, expected gradient vanishes at p=q.
    q = torch.tensor([.05, .25, .5, .8, .95], dtype=torch.float64)
    score = torch.logit(q).requires_grad_()
    zp = torch.stack((score, score * 0, score * 0, score * 0), dim=1)
    cfg = dict(event_loss="proper_confidence", epsilon=epsilon)
    neg = classification_loss(zp, torch.zeros(5, dtype=torch.long), cfg, reduction="none")
    pos = classification_loss(zp, torch.ones(5, dtype=torch.long), cfg, reduction="none")
    expected_risk = ((1 - q) * neg + q * pos).sum()
    torch.testing.assert_close(torch.autograd.grad(expected_risk, score)[0], torch.zeros_like(q),
                               atol=1e-14, rtol=0)


@pytest.mark.parametrize("formulation", ["flat", "factorized"])
def test_all_unknown_is_zero_with_zero_gradient(formulation):
    z = torch.randn(3, 4, requires_grad=True)
    loss = classification_loss(z, torch.full((3,), -1), dict(formulation=formulation))
    loss.backward()
    assert loss == 0 and (z.grad == 0).all()


def test_flat_ce_and_unsupported_combinations():
    z = torch.randn(3, 4)
    y = torch.tensor([0, 3, -1])
    torch.testing.assert_close(classification_loss(z, y, {"formulation": "flat"}),
                               F.cross_entropy(z[:2], y[:2]))
    torch.testing.assert_close(joint_probabilities(z, "flat"), z.softmax(1))
    with pytest.raises(ValueError, match="standard CE"):
        classification_loss(z, y, dict(formulation="flat", event_loss="focal"))
    with pytest.raises(ValueError, match="gamma"):
        classification_loss(z, y, dict(event_loss="focal", gamma=-1))
