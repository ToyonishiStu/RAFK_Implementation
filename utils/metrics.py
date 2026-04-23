"""Evaluation metrics: MAE, Chamfer Distance, IoU, Precision, Recall, F1.

CPU path: scipy.spatial.KDTree with shared trees (workers=-1 for parallel queries).
GPU path: torch.cdist with shared chunked distance computation.

Key optimisation: KDTree / min-distance tables are computed ONCE per point-cloud
pair and reused for Chamfer Distance, P/R/F1, and IoU — avoiding redundant builds.
"""

import numpy as np
import torch
from scipy.spatial import KDTree


# =============================================================================
# CPU helpers
# =============================================================================

def _nn_distances(pred_pts: np.ndarray, gt_pts: np.ndarray):
    """Build KDTrees once and return bidirectional min distances.

    Returns:
        d_p2g: (N,) min L2 distance from each pred point to nearest gt
        d_g2p: (M,) min L2 distance from each gt point to nearest pred
    """
    tree_gt   = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)
    d_p2g, _ = tree_gt.query(pred_pts,   workers=-1)
    d_g2p, _ = tree_pred.query(gt_pts,   workers=-1)
    return d_p2g, d_g2p


def _iou_from_voxels(pred_pts: np.ndarray, gt_pts: np.ndarray,
                     voxel_size: float = 0.1) -> float:
    """Voxel IoU using numpy integer encoding (no Python sets)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0
    inv = 1.0 / voxel_size
    pv  = (np.floor(pred_pts * inv)).astype(np.int64)
    gv  = (np.floor(gt_pts   * inv)).astype(np.int64)
    S, G = 1000, 2001
    def _enc(v):
        vs = v + S
        return vs[:, 0] * G * G + vs[:, 1] * G + vs[:, 2]
    ph  = np.unique(_enc(pv))
    gh  = np.unique(_enc(gv))
    inter = len(np.intersect1d(ph, gh, assume_unique=True))
    union = len(ph) + len(gh) - inter
    return inter / max(union, 1)


# =============================================================================
# Public CPU API  (same signatures as before)
# =============================================================================

def compute_mae(pred_range: np.ndarray, gt_range: np.ndarray,
                mask: np.ndarray) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return 0.0
    pred_m = np.expm1(pred_range[valid])
    gt_m   = np.expm1(gt_range[valid])
    return float(np.abs(pred_m - gt_m).mean())


def compute_chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")
    d_p2g, d_g2p = _nn_distances(pred_pts, gt_pts)
    return float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))


def compute_iou(pred_pts: np.ndarray, gt_pts: np.ndarray,
                voxel_size: float = 0.1) -> float:
    return _iou_from_voxels(pred_pts, gt_pts, voxel_size)


def compute_precision_recall_f1(pred_pts: np.ndarray, gt_pts: np.ndarray,
                                threshold: float = 0.1) -> tuple:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0, 0.0, 0.0
    d_p2g, d_g2p = _nn_distances(pred_pts, gt_pts)
    precision = float(np.mean(d_p2g < threshold))
    recall    = float(np.mean(d_g2p < threshold))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def compute_all_metrics(pred_range: np.ndarray, gt_range: np.ndarray,
                        mask: np.ndarray,
                        pred_pts: np.ndarray, gt_pts: np.ndarray,
                        voxel_size: float = 0.1,
                        threshold: float = 0.1) -> dict:
    """Compute all metrics with shared KDTree (CD + P/R/F1 from one build)."""
    mae = compute_mae(pred_range, gt_range, mask)
    iou = _iou_from_voxels(pred_pts, gt_pts, voxel_size)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"mae": mae, "chamfer_distance": float("inf"),
                "iou": iou, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    d_p2g, d_g2p = _nn_distances(pred_pts, gt_pts)
    cd        = float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))
    precision = float(np.mean(d_p2g < threshold))
    recall    = float(np.mean(d_g2p < threshold))
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"mae": mae, "chamfer_distance": cd, "iou": iou,
            "precision": precision, "recall": recall, "f1": f1}


def compute_mae_by_distance(pred_range: np.ndarray, gt_range: np.ndarray,
                            mask: np.ndarray, config=None,
                            distance_ranges: list = None) -> dict:
    if distance_ranges is None:
        distance_ranges = [(0, 30), (30, 60)]
    valid  = mask > 0
    gt_m   = np.expm1(gt_range)
    pred_m = np.expm1(pred_range)
    results = {}
    for dmin, dmax in distance_ranges:
        label = f"{dmin}-{dmax}m"
        sel   = valid & (gt_m >= dmin) & (gt_m < dmax)
        results[label] = float(np.abs(pred_m[sel] - gt_m[sel]).mean()) if sel.sum() > 0 else float("nan")
    return results


def compute_metrics_by_distance(pred_pts: np.ndarray, gt_pts: np.ndarray,
                                 distance_ranges: list = None) -> dict:
    """Compute per-range metrics with one shared KDTree per subset."""
    if distance_ranges is None:
        distance_ranges = [(0, 10), (10, 30), (30, 50), (50, 80)]

    pred_r = np.linalg.norm(pred_pts, axis=1)
    gt_r   = np.linalg.norm(gt_pts,   axis=1)

    results = {}
    for dmin, dmax in distance_ranges:
        label    = f"{dmin}-{dmax}m"
        pred_sub = pred_pts[(pred_r >= dmin) & (pred_r < dmax)]
        gt_sub   = gt_pts  [(gt_r   >= dmin) & (gt_r   < dmax)]
        if len(pred_sub) > 0 and len(gt_sub) > 0:
            d_p2g, d_g2p = _nn_distances(pred_sub, gt_sub)
            cd        = float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))
            iou       = _iou_from_voxels(pred_sub, gt_sub)
            precision = float(np.mean(d_p2g < 0.1))
            recall    = float(np.mean(d_g2p < 0.1))
            f1        = 2 * precision * recall / max(precision + recall, 1e-8)
            results[label] = {"cd": cd, "iou": iou, "precision": precision,
                              "recall": recall, "f1": f1,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
        else:
            results[label] = {"cd": float("inf"), "iou": 0.0, "precision": 0.0,
                              "recall": 0.0, "f1": 0.0,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
    return results


# =============================================================================
# GPU helpers (torch.cdist, chunked to bound VRAM)
# =============================================================================

def _nn_dists_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                  chunk_size: int = 1024):
    """Bidirectional min-L2 distances via chunked torch.cdist.

    chunk_size=1024 → peak VRAM ≈ 1024 × |ref| × 4 B  (safe on 4 GB GPU).
    Returns (d_p2g, d_g2p) as GPU float32 tensors.
    """
    def _one_way(query, ref):
        mins = []
        for i in range(0, len(query), chunk_size):
            chunk = query[i:i + chunk_size]
            mins.append(torch.cdist(chunk, ref).min(dim=1).values)
        return torch.cat(mins)

    return _one_way(pred_pts, gt_pts), _one_way(gt_pts, pred_pts)


# =============================================================================
# Public GPU API
# =============================================================================

def compute_chamfer_distance_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                                  chunk_size: int = 1024) -> float:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")
    d_p2g, d_g2p = _nn_dists_gpu(pred_pts, gt_pts, chunk_size)
    return ((d_p2g ** 2).mean() + (d_g2p ** 2).mean()).item()


def compute_precision_recall_f1_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                                     threshold: float = 0.1,
                                     chunk_size: int = 1024) -> tuple:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0, 0.0, 0.0
    d_p2g, d_g2p = _nn_dists_gpu(pred_pts, gt_pts, chunk_size)
    precision = (d_p2g < threshold).float().mean().item()
    recall    = (d_g2p < threshold).float().mean().item()
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def compute_iou_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                    voxel_size: float = 0.1) -> float:
    """Voxel IoU using torch.unique + torch.isin (no Python sets)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0
    inv = 1.0 / voxel_size
    pv  = torch.floor(pred_pts * inv).long()
    gv  = torch.floor(gt_pts   * inv).long()
    S, G = 1000, 2001

    def _enc(v):
        vs = v + S
        return vs[:, 0] * G * G + vs[:, 1] * G + vs[:, 2]

    ph  = torch.unique(_enc(pv))
    gh  = torch.unique(_enc(gv))
    inter = torch.isin(ph, gh).sum().item()
    union = len(ph) + len(gh) - inter
    return inter / max(union, 1)


def compute_mae_gpu(pred_range: torch.Tensor, gt_range: torch.Tensor,
                    mask: torch.Tensor) -> float:
    valid = mask > 0
    if valid.sum() == 0:
        return 0.0
    return (torch.expm1(pred_range[valid]) - torch.expm1(gt_range[valid])).abs().mean().item()


def compute_all_metrics_gpu(pred_range: torch.Tensor, gt_range: torch.Tensor,
                             mask: torch.Tensor,
                             pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                             voxel_size: float = 0.1,
                             threshold: float = 0.1,
                             chunk_size: int = 1024) -> dict:
    """GPU metrics with shared distance computation (CD + P/R/F1 one pass)."""
    mae = compute_mae_gpu(pred_range, gt_range, mask)
    iou = compute_iou_gpu(pred_pts, gt_pts, voxel_size)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"mae": mae, "chamfer_distance": float("inf"),
                "iou": iou, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    d_p2g, d_g2p = _nn_dists_gpu(pred_pts, gt_pts, chunk_size)
    cd        = ((d_p2g ** 2).mean() + (d_g2p ** 2).mean()).item()
    precision = (d_p2g < threshold).float().mean().item()
    recall    = (d_g2p < threshold).float().mean().item()
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"mae": mae, "chamfer_distance": cd, "iou": iou,
            "precision": precision, "recall": recall, "f1": f1}


def compute_mae_by_distance_gpu(pred_range: torch.Tensor, gt_range: torch.Tensor,
                                 mask: torch.Tensor,
                                 distance_ranges: list = None) -> dict:
    if distance_ranges is None:
        distance_ranges = [(0, 30), (30, 60)]
    valid  = mask > 0
    gt_m   = torch.expm1(gt_range)
    pred_m = torch.expm1(pred_range)
    results = {}
    for dmin, dmax in distance_ranges:
        label = f"{dmin}-{dmax}m"
        sel   = valid & (gt_m >= dmin) & (gt_m < dmax)
        results[label] = (pred_m[sel] - gt_m[sel]).abs().mean().item() if sel.sum() > 0 else float("nan")
    return results


def compute_metrics_by_distance_gpu(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                                     distance_ranges: list = None,
                                     chunk_size: int = 1024) -> dict:
    """GPU per-range metrics with shared min-distance per subset."""
    if distance_ranges is None:
        distance_ranges = [(0, 10), (10, 30), (30, 50), (50, 80)]

    pred_r = torch.norm(pred_pts, dim=1)
    gt_r   = torch.norm(gt_pts,   dim=1)

    results = {}
    for dmin, dmax in distance_ranges:
        label    = f"{dmin}-{dmax}m"
        pred_sub = pred_pts[(pred_r >= dmin) & (pred_r < dmax)]
        gt_sub   = gt_pts  [(gt_r   >= dmin) & (gt_r   < dmax)]
        if len(pred_sub) > 0 and len(gt_sub) > 0:
            d_p2g, d_g2p = _nn_dists_gpu(pred_sub, gt_sub, chunk_size)
            cd        = ((d_p2g ** 2).mean() + (d_g2p ** 2).mean()).item()
            iou       = compute_iou_gpu(pred_sub, gt_sub)
            precision = (d_p2g < 0.1).float().mean().item()
            recall    = (d_g2p < 0.1).float().mean().item()
            f1        = 2 * precision * recall / max(precision + recall, 1e-8)
            results[label] = {"cd": cd, "iou": iou, "precision": precision,
                              "recall": recall, "f1": f1,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
        else:
            results[label] = {"cd": float("inf"), "iou": 0.0, "precision": 0.0,
                              "recall": 0.0, "f1": 0.0,
                              "num_pred": len(pred_sub), "num_gt": len(gt_sub)}
    return results
