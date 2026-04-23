"""Range image -> 3D point cloud reprojection."""

import numpy as np
import math
import torch
from config.default import Config


def range_image_to_points(
    range_img: np.ndarray,
    mask: np.ndarray = None,
    config: Config = None,
) -> np.ndarray:
    """
    Convert a log-compressed range image back to 3D points.

    Args:
        range_img: (H, W) log-compressed range image
        mask: (H, W) validity mask. If None, derived from range_img > 0.
        config: Config for FOV parameters.

    Returns:
        points: (N, 3) array [x, y, z]
    """
    if config is None:
        config = Config()

    H, W = range_img.shape
    if mask is None:
        mask = (range_img > 0).astype(np.float32)

    # Undo log compression
    r = np.expm1(range_img)  # exp(x) - 1

    # Build pitch/yaw grids
    rows = np.arange(H, dtype=np.float32)
    cols = np.arange(W, dtype=np.float32)
    col_grid, row_grid = np.meshgrid(cols, rows)

    pitch = config.fov_up_rad - (row_grid / H) * config.fov_total_rad
    yaw = np.pi - (col_grid / W) * 2.0 * np.pi

    # 3D coordinates
    cos_pitch = np.cos(pitch)
    x = r * cos_pitch * np.cos(yaw)
    y = r * cos_pitch * np.sin(yaw)
    z = r * np.sin(pitch)

    # Filter valid points
    valid = mask > 0
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points


def range_image_to_points_gpu(
    range_t: torch.Tensor,
    mask_t: torch.Tensor,
    config: Config = None,
) -> torch.Tensor:
    """GPU-accelerated range image -> 3D point cloud.

    Args:
        range_t: (H, W) log-compressed range image, GPU float32 tensor
        mask_t:  (H, W) validity mask, GPU float32 tensor
        config:  Config for FOV parameters

    Returns:
        points: (N, 3) GPU float32 tensor [x, y, z]
    """
    if config is None:
        config = Config()

    H, W = range_t.shape
    device = range_t.device

    r = torch.expm1(range_t)

    rows = torch.arange(H, dtype=torch.float32, device=device)
    cols = torch.arange(W, dtype=torch.float32, device=device)
    col_grid, row_grid = torch.meshgrid(cols, rows, indexing="xy")

    pitch = config.fov_up_rad - (row_grid / H) * config.fov_total_rad
    yaw   = math.pi - (col_grid / W) * 2.0 * math.pi

    cos_pitch = torch.cos(pitch)
    x = r * cos_pitch * torch.cos(yaw)
    y = r * cos_pitch * torch.sin(yaw)
    z = r * torch.sin(pitch)

    valid = mask_t > 0
    return torch.stack([x[valid], y[valid], z[valid]], dim=-1)
