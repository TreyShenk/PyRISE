import nox

# Use uv to create virtualenvs — it can auto-download Python versions as needed.
nox.options.default_venv_backend = "uv"

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

SMOKE_TEST = """
import sys
import numpy as np
from pyrise import (
    RISE, BoundingBox, sg_noise_floor, arpls_noise_floor, plot_detections,
    iou, iou_matrix, detection_metrics, metrics_curve, EvalResult,
)

# Synthetic TF plot with one injected signal block
rng = np.random.default_rng(0)
tf = rng.exponential(1.0, size=(100, 64))
tf[30:60, 20:40] += 20.0

# Default SG noise floor
boxes_sg = RISE(psd_offset=1.0).run(tf)
assert len(boxes_sg) >= 1, f"SG: expected >=1 detection, got {len(boxes_sg)}"

# arPLS noise floor
boxes_arpls = RISE(noise_floor_fn=arpls_noise_floor, psd_offset=1.0).run(tf)
assert len(boxes_arpls) >= 1, f"arPLS: expected >=1 detection, got {len(boxes_arpls)}"

# BoundingBox is frozen
b = boxes_sg[0]
try:
    b.t0 = 0
    raise AssertionError("BoundingBox should be frozen")
except AttributeError:
    pass

# Metrics — ground truth matches the injected block
gt = [BoundingBox(t0=30, t1=59, f0=20, f1=39)]

# iou: perfect self-overlap should be 1.0
assert iou(gt[0], gt[0]) == 1.0, "self-IoU must be 1.0"
assert iou(gt[0], BoundingBox(0, 5, 0, 5)) == 0.0, "non-overlapping IoU must be 0.0"

# detection_metrics at loose threshold should detect the signal
result = detection_metrics(gt, boxes_sg, iou_threshold=0.3)
assert isinstance(result, EvalResult)
assert result.pd == 1.0, f"expected pd=1.0, got {result.pd}"
assert 0.0 <= result.pfa <= 1.0
assert 0.0 <= result.mean_iou <= 1.0
assert 0.0 <= result.f1 <= 1.0

# metrics_curve returns consistent-length arrays
curve = metrics_curve(gt, boxes_sg)
n = len(curve["thresholds"])
assert all(len(curve[k]) == n for k in ("pd", "pfa", "mean_iou", "f1"))

# edge cases: no detections → pd=0, pfa=0; no GT → pd=0, pfa=1
r_no_det = detection_metrics(gt, [])
assert r_no_det.pd == 0.0 and r_no_det.pfa == 0.0
r_no_gt = detection_metrics([], boxes_sg)
assert r_no_gt.pd == 0.0 and r_no_gt.pfa == 1.0

print(f"Python {sys.version[:6]}  SG={len(boxes_sg)} box(es)  arPLS={len(boxes_arpls)} box(es)  pd={result.pd:.2f}  pfa={result.pfa:.2f}  f1={result.f1:.2f}  OK")
"""


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Smoke-test pyrise against each supported Python version."""
    session.install(".[arpls]")
    session.run("python", "-c", SMOKE_TEST)
