from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyrise.pipeline import BoundingBox


def plot_detections(
    tf_plot: np.ndarray,
    boxes: list[BoundingBox],
    *,
    ax: plt.Axes | None = None,
    db: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    box_color: str = "red",
    box_linewidth: float = 1.5,
    fs: float | None = None,
    fft_size: int | None = None,
    title: str = "RISE Detections",
) -> plt.Figure:
    """Plot a time-frequency spectrogram with detected bounding boxes overlaid.

    Parameters
    ----------
    tf_plot:
        2-D array of shape (T, F) in **linear power**. Time is axis 0,
        frequency is axis 1 — the same convention as :meth:`RISE.run`.
    boxes:
        Bounding boxes returned by :meth:`RISE.run`.
    ax:
        Existing ``Axes`` to draw on. If ``None``, a new figure and axes are
        created and returned.
    db:
        If ``True`` (default), convert the TF plot to dB before display:
        ``10 * log10(tf_plot)``. Pass ``False`` to display in linear power.
    vmin, vmax:
        Color scale limits. Passed directly to ``imshow``. If ``None``,
        matplotlib chooses them automatically.
    cmap:
        Colormap name for the spectrogram image.
    box_color:
        Edge color for the bounding box rectangles.
    box_linewidth:
        Line width for the bounding box rectangles.
    fs:
        Sampling rate in Hz. If provided together with ``fft_size``, the
        frequency axis is labeled in MHz and the time axis in milliseconds.
    fft_size:
        FFT size (number of frequency bins). Required alongside ``fs`` to
        compute physical axis labels.
    title:
        Axes title.

    Returns
    -------
    plt.Figure
        The figure containing the plot.
    """
    display = 10 * np.log10(np.maximum(tf_plot, 1e-12)) if db else tf_plot
    units = "dB" if db else "linear"

    use_physical = fs is not None and fft_size is not None
    T, F = tf_plot.shape
    if use_physical:
        dt = fft_size / fs                    # seconds per time bin
        df = fs / fft_size                    # Hz per frequency bin
        extent = [0, F * df / 1e6, T * dt * 1e3, 0]  # [f_min MHz, f_max MHz, t_max ms, t_min ms]
        xlabel = "Frequency (MHz)"
        ylabel = "Time (ms)"
    else:
        extent = [0, F, T, 0]
        xlabel = "Frequency (bin)"
        ylabel = "Time (bin)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        display,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=f"Power ({units})", pad=0.02)

    for box in boxes:
        if use_physical:
            x0 = box.f0 * df / 1e6
            y0 = box.t0 * dt * 1e3
            width = (box.f1 - box.f0 + 1) * df / 1e6
            height = (box.t1 - box.t0 + 1) * dt * 1e3
        else:
            x0 = box.f0
            y0 = box.t0
            width = box.f1 - box.f0 + 1
            height = box.t1 - box.t0 + 1

        rect = mpatches.Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=box_linewidth,
            edgecolor=box_color,
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig
