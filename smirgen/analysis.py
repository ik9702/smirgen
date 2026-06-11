"""Post-processing helpers for SMIR transfer functions.

Express the transfer function magnitude in dB against a chosen reference:

* ``ref="source"`` (A) - relative to the source's free-field level at a
  reference distance (1 m by default). Since the generator already uses the
  ``1/R`` free-field convention, this is simply ``20*log10(|H| * ref_distance)``;
  0 dB == "what the source produces at ``ref_distance`` in free field".
* ``ref="direct"`` (B) - relative to the direct sound actually picked up by the
  array (distance + array colouration removed). 0 dB == the direct arrival, so
  only the room colouration / reverberation remains.
"""

import numpy as np

from .generator import smir_generator

__all__ = ["relative_db", "direct_reference"]


def direct_reference(**kwargs):
    """Return the direct-path-only transfer function for the same setup.

    Re-runs :func:`smir_generator` with perfectly absorbing walls
    (``beta = 0``) and ``order = 0`` so only the direct sound remains, keeping
    the array, radius, ``N_harm``, geometry, sampling and source directivity
    identical. Accepts the same keyword arguments as :func:`smir_generator`.
    """
    ref = dict(kwargs)
    ref.update(beta=[0, 0, 0, 0, 0, 0], order=0, refl_coeff_ang_dep=0, HP=0,
               plot_geometry=False)
    _, H_dir, _ = smir_generator(**ref)
    return H_dir


def relative_db(ref="source", ref_distance=1.0, eps=1e-12, **kwargs):
    """Transfer-function magnitude in dB, relative to a chosen reference.

    The returned ``H`` is the pre-high-pass transfer function (the optional HP
    filter only affects the time-domain RIR), so the result is independent of
    the ``HP`` flag.

    Parameters
    ----------
    ref : {"source", "direct"}
        ``"source"`` (A): 0 dB is the source's free-field level at
        ``ref_distance``. ``"direct"`` (B): 0 dB is the direct arrival at the
        array (an extra direct-path-only run is used as the reference).
    ref_distance : float
        Reference distance in metres for ``ref="source"`` (default 1 m).
    eps : float
        Small floor to avoid ``log10(0)``.
    **kwargs
        Same keyword arguments as :func:`smir_generator` (pass as keywords).

    Returns
    -------
    freq : (F,) ndarray
        One-sided frequency axis in Hz.
    db : (M, F) ndarray
        Magnitude in dB relative to the chosen reference. Bins where the
        magnitude (or the reference) is exactly zero are set to NaN.
    H : (M, F) ndarray (complex)
        The full transfer function, for reference.
    """
    if ref not in ("source", "direct"):
        raise ValueError("ref must be 'source' or 'direct'.")

    h, H, _ = smir_generator(**kwargs)
    nsample = h.shape[1]
    K = kwargs.get("K", 2)
    procFs = kwargs["procFs"]
    freq = np.arange(H.shape[1]) * procFs / (K * nsample)

    with np.errstate(divide="ignore", invalid="ignore"):
        if ref == "source":
            # H is already in the source's 1/R free-field convention, so the
            # level relative to the source at `ref_distance` is |H| * d_ref.
            db = 20.0 * np.log10(np.abs(H) * ref_distance + eps)
            db[np.abs(H) == 0] = np.nan          # zeroed DC bin for arrays
        else:  # direct
            ref_kwargs = dict(kwargs)
            ref_kwargs["nsample"] = nsample
            H_dir = direct_reference(**ref_kwargs)
            db = 20.0 * np.log10((np.abs(H) + eps) / (np.abs(H_dir) + eps))
            db[:, 0] = np.nan                    # 0 Hz reference is zero

    return freq, db, H
