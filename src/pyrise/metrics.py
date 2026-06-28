from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrise.pipeline import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute intersection-over-union of two bounding boxes.

    All BoundingBox indices are inclusive, so a single-bin box has area 1.

    Returns 0.0 for non-overlapping boxes.
    """
    t0 = max(a.t0, b.t0)
    t1 = min(a.t1, b.t1)
    f0 = max(a.f0, b.f0)
    f1 = min(a.f1, b.f1)

    if t0 > t1 or f0 > f1:
        return 0.0

    intersection = (t1 - t0 + 1) * (f1 - f0 + 1)
    area_a = (a.t1 - a.t0 + 1) * (a.f1 - a.f0 + 1)
    area_b = (b.t1 - b.t0 + 1) * (b.f1 - b.f0 + 1)
    return intersection / (area_a + area_b - intersection)


def iou_matrix(
    gt_boxes: list[BoundingBox],
    pred_boxes: list[BoundingBox],
) -> np.ndarray:
    """Compute the pairwise IoU matrix I from the paper.

    Parameters
    ----------
    gt_boxes:
        Ground-truth bounding boxes. Length Ngt.
    pred_boxes:
        Detected bounding boxes. Length Nd.

    Returns
    -------
    np.ndarray
        Shape ``(Ngt, Nd)``. ``I[i, j] = iou(gt_boxes[i], pred_boxes[j])``.
        Returns shape ``(0, Nd)`` or ``(Ngt, 0)`` when either list is empty.
    """
    ngt = len(gt_boxes)
    nd = len(pred_boxes)
    mat = np.zeros((ngt, nd), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            mat[i, j] = iou(g, p)
    return mat


@dataclass(frozen=True)
class EvalResult:
    """Evaluation metrics at a single IoU threshold.

    Attributes
    ----------
    iou_threshold:
        The θ_IoU value used to compute these metrics.
    pd:
        Probability of detection — ``Nt / Ngt`` (paper eq. 3).
        Fraction of ground-truth signals covered by at least one detection
        with IoU > threshold.
    pfa:
        Probability of false alarm — ``Nf / Nd`` (paper eq. 3).
        Fraction of detections that do not cover any ground-truth signal
        with IoU > threshold.
    mean_iou:
        Mean of ``max_j I[i, j]`` over all ground-truth boxes (misses
        contribute 0). Bounded above by ``pd`` per the paper.
    f1:
        Harmonic mean of precision (``1 - pfa``) and recall (``pd``).
        Useful single-number summary when balancing false alarms and misses.
    n_gt:
        Number of ground-truth boxes.
    n_det:
        Number of detected boxes.
    """

    iou_threshold: float
    pd: float
    pfa: float
    mean_iou: float
    f1: float
    n_gt: int
    n_det: int


def detection_metrics(
    gt_boxes: list[BoundingBox],
    pred_boxes: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> EvalResult:
    """Compute Pd, Pfa, mean IoU, and F1 at a single IoU threshold.

    Implements the paper's matching rule exactly: each ground-truth box
    independently checks whether *any* detection covers it above the threshold,
    and each detection independently checks whether *any* ground-truth box covers
    it. This is not optimal assignment — a single detection can validate multiple
    ground-truth boxes simultaneously.

    Parameters
    ----------
    gt_boxes:
        Ground-truth bounding boxes.
    pred_boxes:
        Detected bounding boxes from :meth:`RISE.run`.
    iou_threshold:
        IoU threshold θ_IoU. Boxes with IoU strictly above this value are
        considered matched. The paper reports results at 0.4 and 0.5.

    Returns
    -------
    EvalResult
    """
    ngt = len(gt_boxes)
    nd = len(pred_boxes)

    if ngt == 0 and nd == 0:
        return EvalResult(iou_threshold=iou_threshold, pd=0.0, pfa=0.0,
                          mean_iou=0.0, f1=0.0, n_gt=0, n_det=0)

    if ngt == 0:
        return EvalResult(iou_threshold=iou_threshold, pd=0.0, pfa=1.0,
                          mean_iou=0.0, f1=0.0, n_gt=0, n_det=nd)

    if nd == 0:
        return EvalResult(iou_threshold=iou_threshold, pd=0.0, pfa=0.0,
                          mean_iou=0.0, f1=0.0, n_gt=ngt, n_det=0)

    I = iou_matrix(gt_boxes, pred_boxes)

    # Nt: GT boxes covered by at least one detection above threshold
    nt = int(np.sum(I.max(axis=1) > iou_threshold))
    # Nf: detections not covering any GT box above threshold
    nf = int(np.sum(I.max(axis=0) < iou_threshold))

    pd = nt / ngt
    pfa = nf / nd
    mean_iou = float(I.max(axis=1).mean())

    precision = 1.0 - pfa
    f1 = (2 * precision * pd / (precision + pd)) if (precision + pd) > 0 else 0.0

    return EvalResult(
        iou_threshold=iou_threshold,
        pd=pd,
        pfa=pfa,
        mean_iou=mean_iou,
        f1=f1,
        n_gt=ngt,
        n_det=nd,
    )


def metrics_curve(
    gt_boxes: list[BoundingBox],
    pred_boxes: list[BoundingBox],
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute Pd, Pfa, mean IoU, and F1 across a range of IoU thresholds.

    The IoU matrix is computed once and reused across all thresholds.

    Parameters
    ----------
    gt_boxes:
        Ground-truth bounding boxes.
    pred_boxes:
        Detected bounding boxes from :meth:`RISE.run`.
    thresholds:
        1-D array of IoU threshold values. Defaults to
        ``np.linspace(0.1, 0.9, 17)``.

    Returns
    -------
    dict with keys ``"thresholds"``, ``"pd"``, ``"pfa"``, ``"mean_iou"``,
    ``"f1"`` — all 1-D arrays of the same length. Pass directly to
    :func:`~pyrise.vis.plot_metrics_curve`.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    results = [detection_metrics(gt_boxes, pred_boxes, float(t)) for t in thresholds]

    return {
        "thresholds": np.asarray(thresholds),
        "pd":         np.array([r.pd        for r in results]),
        "pfa":        np.array([r.pfa       for r in results]),
        "mean_iou":   np.array([r.mean_iou  for r in results]),
        "f1":         np.array([r.f1        for r in results]),
    }
