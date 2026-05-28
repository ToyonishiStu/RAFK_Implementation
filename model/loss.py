"""Loss functions for FLASH / FLASH+ super-resolution."""

import torch
import torch.nn.functional as F


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """L1 loss masked to valid pixels only.
    Args:
        pred: (B, 1, H, W) predicted range image
        target: (B, 1, H, W) ground truth range image
        mask: (B, 1, H, W) validity mask (1=valid, 0=invalid)
    Returns:
        scalar loss
    """
    loss = torch.abs(pred - target) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def freq_consistency_loss(weight_pairs: list) -> torch.Tensor:
    """Frequency consistency loss: encourage near/far filters to differ.

    L_freq = -sum(||W_near - W_far||_F) / num_pairs

    Args:
        weight_pairs: list of (W_near, W_far) weight tensors
    Returns:
        scalar loss (negative → minimizing encourages divergence)
    """
    if not weight_pairs:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=weight_pairs[0][0].device)
    for w_near, w_far in weight_pairs:
        total = total - torch.norm(w_near - w_far, p="fro")
    return total / len(weight_pairs)
