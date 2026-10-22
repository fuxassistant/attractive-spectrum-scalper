"""Masked losses and coherent probabilities for NONE/L_ONLY/S_ONLY/BOTH.

Factorized logits are [event, conditional_long, conditional_short,
conditional_both]. The event is *target*-gated, never prediction-gated.
Every mean uses the same number of known roots for event and direction terms.
Unknown y=-1 contributes exactly zero loss and gradient. Cohort selection is
the data adapter's responsibility; partial positives must not silently be
pooled with only fully mature negatives.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _check_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != 4 or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor with shape [B,4]")


def joint_probabilities(logits: torch.Tensor, formulation: str = "factorized") -> torch.Tensor:
    """Return a normalized [B,4] distribution in NONE,L_ONLY,S_ONLY,BOTH order.

    With non-proper focal loss these are coherent scores, not guaranteed
    calibrated probabilities. Focal's gamma=0 recovers the BCE objective.
    """
    _check_logits(logits)
    if formulation == "flat":
        return logits.softmax(dim=-1)
    if formulation != "factorized":
        raise ValueError(f"Unsupported formulation {formulation!r}")
    event = logits[:, :1].sigmoid()
    direction = logits[:, 1:].softmax(dim=-1)
    return torch.cat(((-logits[:, :1]).sigmoid(), event * direction), dim=-1)


def classification_loss(
    logits: torch.Tensor, y: torch.Tensor, config: dict, reduction: str = "mean",
) -> torch.Tensor:
    """Compute masked loss; reduction='none' returns B unnormalized root losses.

    Supported event losses: BCE, focal (gamma >= 0), BCE+Brier (lambda >= 0),
    proper confidence BCE - 2*(1-epsilon)*(p-y)^2 with 0 < epsilon <= 1.
    The latter has gradient (p-y)[epsilon+(1-epsilon)(2p-1)^2].
    Flat formulation intentionally supports only standard four-class CE.
    """
    _check_logits(logits)
    if y.ndim != 1 or len(y) != len(logits) or y.dtype != torch.long:
        raise ValueError("y must be an int64 tensor with shape [B]")
    if y.device != logits.device:
        raise ValueError("y and logits must be on the same device")
    if ((y < -1) | (y > 3)).any().item():
        raise ValueError("y must contain only -1, 0, 1, 2, 3")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    formulation = config.get("formulation", "factorized")
    event_loss = config.get("event_loss", "bce")
    known = y >= 0
    if formulation == "flat":
        if event_loss != "bce":
            raise ValueError("Flat formulation supports only standard CE (event_loss='bce')")
        per_root = F.cross_entropy(logits, y.clamp_min(0), reduction="none")
    elif formulation == "factorized":
        event_y = (y > 0).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits[:, 0], event_y, reduction="none")
        p = logits[:, 0].sigmoid()
        if event_loss == "bce":
            event_term = bce
        elif event_loss == "focal":
            gamma = float(config.get("gamma", 1.0))
            if not math.isfinite(gamma) or gamma < 0:
                raise ValueError("gamma must be finite and >= 0")
            # For gamma < 1, pow'(0) is infinite even though the complete
            # focal derivative tends to zero. Guard only machine underflow.
            difficulty = (-torch.expm1(-bce)).clamp_min(torch.finfo(bce.dtype).tiny)
            event_term = difficulty.pow(gamma) * bce
        elif event_loss == "bce_brier":
            weight = float(config.get("brier_lambda", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("brier_lambda must be finite and >= 0")
            event_term = bce + weight * (p - event_y).square()
        elif event_loss == "proper_confidence":
            epsilon = float(config.get("epsilon", 1 / 3))
            if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
                raise ValueError("epsilon must be finite and in (0, 1]")
            event_term = bce - 2 * (1 - epsilon) * (p - event_y).square()
        else:
            raise ValueError(f"Unsupported event loss {event_loss!r}")
        direction = F.cross_entropy(logits[:, 1:], (y - 1).clamp_min(0), reduction="none")
        per_root = event_term + torch.where(y > 0, direction, torch.zeros_like(direction))
    else:
        raise ValueError(f"Unsupported formulation {formulation!r}")
    per_root = torch.where(known, per_root, torch.zeros_like(per_root))
    if reduction == "none":
        return per_root
    total = per_root.sum()
    if reduction == "sum":
        return total
    return total / known.sum().clamp_min(1)
