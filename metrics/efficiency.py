"""Machine-independent efficiency metrics for comparing submissions."""

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return the total and the trainable parameter counts.

    Grading uses the total. Freezing layers changes what you train, not how large the
    model you ship is, so a frozen backbone still costs its parameters.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def measure_gflops(model: nn.Module, sample_CHW: Tensor, device: torch.device) -> float:
    """Return billions of floating-point operations for one forward pass on ``sample_CHW``.

    The count comes from tracing tensor operations rather than from timing anything, so
    the same model and image give the same number on every machine.

    The image matters: a detector drops region proposals whose objectness falls below
    ``rpn_score_thresh``, so how much work the box head does depends on what the model
    sees. Callers pass a fixed validation image to keep submissions comparable.
    """
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model([sample_CHW.to(device)])
    return counter.get_total_flops() / 1e9
