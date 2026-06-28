from typing import Callable

import numpy as np
from scipy.signal import argrelmin, savgol_filter

NoiseFloorFn = Callable[[np.ndarray], np.ndarray]
"""
Callable[[psd], noise_floor]

Takes a PSD array of shape (F,) in **linear power** and returns a noise floor
array of the same shape in linear power. The RISE pipeline uses the returned
values to prune frequency columns: any column whose PSD value falls below
``noise_floor[f] + psd_offset`` is skipped during binarization.
"""


def sg_noise_floor(
    psd: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Estimate a global noise floor via a Savitzky-Golay smoothed PSD.

    Smooths the PSD, finds the lowest local minimum, and returns that scalar
    broadcast to the full frequency axis — the same approach used in the RISE
    paper.

    Parameters
    ----------
    psd:
        1-D array of shape (F,) in linear power, aggregated over the time axis.
    window_length:
        Length of the SG filter window. Must be odd and greater than
        ``polyorder``. Larger values produce smoother baselines.
    polyorder:
        Polynomial order for the SG filter. Must be less than ``window_length``.

    Returns
    -------
    np.ndarray
        Constant array of shape (F,) — every element equals the estimated
        global noise floor N0.
    """
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length ({window_length}) must be greater than polyorder ({polyorder})"
        )

    smoothed = savgol_filter(psd, window_length, polyorder)
    (minima_idx,) = argrelmin(smoothed, order=1)

    if minima_idx.size == 0:
        n0 = float(smoothed.min())
    else:
        n0 = float(smoothed[minima_idx].min())

    return np.full(psd.shape, n0, dtype=psd.dtype)


def arpls_noise_floor(
    psd: np.ndarray,
    *,
    lam: float = 1e5,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """Estimate a per-frequency noise floor via arPLS (pybaselines).

    Unlike :func:`sg_noise_floor`, this returns a *per-frequency* baseline
    rather than a global scalar, which can better track a frequency-dependent
    noise floor (e.g., roll-off near band edges).

    Requires ``pybaselines`` — install with::

        uv add pybaselines

    Parameters
    ----------
    psd:
        1-D array of shape (F,) in linear power.
    lam:
        Smoothness regularization. Larger values → smoother, flatter baseline.
    tol:
        Convergence tolerance.
    max_iter:
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        Per-frequency noise floor of shape (F,), same dtype as ``psd``.
    """
    try:
        from pybaselines.whittaker import arpls
    except ImportError as exc:
        raise ImportError(
            "pybaselines is required for arpls_noise_floor. "
            "Install it with: uv add pybaselines"
        ) from exc

    baseline, _ = arpls(psd, lam=lam, tol=tol, max_iter=max_iter)
    return baseline.astype(psd.dtype)
