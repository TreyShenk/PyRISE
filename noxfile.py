import nox

# Use uv to create virtualenvs — it can auto-download Python versions as needed.
nox.options.default_venv_backend = "uv"

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

SMOKE_TEST = """
import sys
import numpy as np
from pyrise import RISE, BoundingBox, sg_noise_floor, arpls_noise_floor, plot_detections

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

print(f"Python {sys.version[:6]}  SG={len(boxes_sg)} box(es)  arPLS={len(boxes_arpls)} box(es)  OK")
"""


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Smoke-test pyrise against each supported Python version."""
    session.install(".[arpls]")
    session.run("python", "-c", SMOKE_TEST)
