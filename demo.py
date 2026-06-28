"""
RISE Demo — Synthetic spectrum sensing.

Builds a noisy time-frequency plot, injects rectangular signal blocks to
simulate transmissions, runs the RISE pipeline, and plots the result.

Run with:
    uv run python demo.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pyrise import RISE, BoundingBox, sg_noise_floor, arpls_noise_floor, plot_detections

# ── Synthetic TF plot ─────────────────────────────────────────────────────────

rng = np.random.default_rng(0)
T, F = 400, 256         # time bins × frequency bins
NOISE_POWER = 1.0       # mean noise power (linear)

# Ground-truth signal blocks: (t0, t1, f0, f1, snr_linear)
SIGNALS = [
    ( 40, 140,  50,  80, 30.0),   # tall, medium bandwidth
    ( 80, 120, 130, 200, 20.0),   # wide, moderate SNR
    (200, 300,  20,  55, 40.0),   # strong, narrower
    (310, 370, 170, 230, 25.0),   # lower-right
]

tf = rng.exponential(NOISE_POWER, size=(T, F))
for t0, t1, f0, f1, snr in SIGNALS:
    tf[t0 : t1 + 1, f0 : f1 + 1] += snr * NOISE_POWER

gt_boxes = [BoundingBox(t0=t0, t1=t1, f0=f0, f1=f1) for t0, t1, f0, f1, _ in SIGNALS]

# ── Run RISE ──────────────────────────────────────────────────────────────────

rise = RISE(noise_floor_fn=sg_noise_floor, psd_offset=1.0)
detected = rise.run(tf)

print(f"Ground truth : {len(gt_boxes)} signal(s)")
print(f"Detected     : {len(detected)} bounding box(es)")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
plot_detections(tf, detected, ax=ax, title="RISE Demo — arPLS noise floor")

for b in gt_boxes:
    ax.add_patch(
        mpatches.Rectangle(
            (b.f0, b.t0),
            b.f1 - b.f0 + 1,
            b.t1 - b.t0 + 1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
    )

ax.legend(
    handles=[
        mpatches.Patch(edgecolor="red",  facecolor="none",              label="Detected"),
        mpatches.Patch(edgecolor="lime", facecolor="none", linestyle="--", label="Ground truth"),
    ],
    loc="upper right",
)

plt.tight_layout()
plt.savefig("demo_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved demo_output.png")
