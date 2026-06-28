from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu

from pyrise.noise import NoiseFloorFn, sg_noise_floor


@dataclass(frozen=True)
class BoundingBox:
    """A detected signal region in bin-index units.

    All indices are inclusive. Convert to physical units with::

        t_seconds  = bbox.t0 * (fft_size / fs)
        f_hz       = bbox.f0 * (fs / fft_size)
    """

    t0: int  # time start bin (inclusive)
    t1: int  # time end bin (inclusive)
    f0: int  # frequency start bin (inclusive)
    f1: int  # frequency end bin (inclusive)


class RISE:
    """Real-time Image processing for Spectral Energy detection and localization.

    A Python port of the RISE algorithm (Tung et al., arXiv:2603.20481).
    Treats a time-frequency spectrogram as an image and applies:

    1. Noise floor estimation + frequency pruning
    2. Adaptive Otsu binarization (per frequency column)
    3. Morphological operations (closing → opening → horizontal opening)
    4. Connected component labeling → bounding boxes

    Parameters
    ----------
    noise_floor_fn:
        Callable with signature ``(psd: np.ndarray) -> np.ndarray``.
        Takes the time-averaged PSD of shape (F,) in linear power and returns
        a noise floor estimate of shape (F,) in linear power. Default is
        :func:`~rise.noise.sg_noise_floor`. Swap in
        :func:`~rise.noise.arpls_noise_floor` or any custom function to change
        the noise floor estimation strategy.
    psd_offset:
        Additive offset (linear power units) added to the noise floor before
        column pruning: ``threshold[f] = noise_floor[f] + psd_offset``.
        Columns whose mean PSD falls below the threshold are skipped during
        binarization. Tune this to control sensitivity — a value of 0.0 prunes
        only columns that fall below the estimated floor.
    se_size_2d:
        Side length of the square 2-D morphological structuring element used
        for closing and opening. Paper uses 3.
    se_size_1d:
        Length of the horizontal 1-D structuring element used for the final
        opening pass (removes thin horizontal streaks). Paper uses 3.

    Examples
    --------
    Default usage::

        import numpy as np
        from rise import RISE

        tf_plot = ...  # shape (T, F), linear power
        rise = RISE()
        boxes = rise.run(tf_plot)

    Swap in arPLS noise floor::

        from functools import partial
        from rise import RISE, arpls_noise_floor

        rise = RISE(noise_floor_fn=arpls_noise_floor)

    Custom noise floor estimator::

        def median_floor(psd):
            return np.full_like(psd, np.median(psd))

        rise = RISE(noise_floor_fn=median_floor)
    """

    def __init__(
        self,
        noise_floor_fn: NoiseFloorFn = sg_noise_floor,
        *,
        psd_offset: float = 0.0,
        se_size_2d: int = 3,
        se_size_1d: int = 3,
    ) -> None:
        self.noise_floor_fn = noise_floor_fn
        self.psd_offset = psd_offset
        self.se_size_2d = se_size_2d
        self.se_size_1d = se_size_1d

    def run(self, tf_plot: np.ndarray) -> list[BoundingBox]:
        """Run the full RISE pipeline on a time-frequency plot.

        Parameters
        ----------
        tf_plot:
            2-D array of shape (T, F) in **linear power** (not dB).
            Axis 0 = time bins, axis 1 = frequency bins. Values must be
            non-negative. Pass ``10 ** (tf_db / 10)`` to convert from dB first.

        Returns
        -------
        list[BoundingBox]
            One :class:`BoundingBox` per detected signal component, in
            bin-index units. Empty list if nothing is detected.
        """
        psd = self._estimate_psd(tf_plot)
        active_mask = self._prune_columns(psd)
        binary = self._binarize(tf_plot, active_mask)
        cleaned = self._morphological_ops(binary)
        return self._extract_bboxes(cleaned)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _estimate_psd(self, tf_plot: np.ndarray) -> np.ndarray:
        return tf_plot.mean(axis=0)

    def _prune_columns(self, psd: np.ndarray) -> np.ndarray:
        noise_floor = self.noise_floor_fn(psd)
        threshold = noise_floor + self.psd_offset
        return psd >= threshold

    def _binarize(self, tf_plot: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        T, F = tf_plot.shape
        binary = np.zeros((T, F), dtype=bool)
        for f in np.where(active_mask)[0]:
            col = tf_plot[:, f]
            if col.max() == col.min():
                continue
            thresh = threshold_otsu(col)
            binary[:, f] = col > thresh
        return binary

    def _morphological_ops(self, binary: np.ndarray) -> np.ndarray:
        se_2d = np.ones((self.se_size_2d, self.se_size_2d), dtype=bool)
        se_1d = np.ones((1, self.se_size_1d), dtype=bool)

        # Order matches the paper: close → open → horizontal open
        result = ndi.binary_closing(binary, structure=se_2d)
        result = ndi.binary_opening(result, structure=se_2d)
        result = ndi.binary_opening(result, structure=se_1d)
        return result

    def _extract_bboxes(self, binary: np.ndarray) -> list[BoundingBox]:
        # 4-connectivity (cross structure) matches the paper
        labeled, _ = ndi.label(binary)
        slices = ndi.find_objects(labeled)
        boxes: list[BoundingBox] = []
        for s in slices:
            if s is None:
                continue
            boxes.append(
                BoundingBox(
                    t0=s[0].start,
                    t1=s[0].stop - 1,
                    f0=s[1].start,
                    f1=s[1].stop - 1,
                )
            )
        return boxes
