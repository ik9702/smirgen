"""Python port of ``smir_generator.m``.

Generates room impulse responses between a source and a spherical microphone
array (open or rigid) in a reverberant shoebox enclosure, using the image
method combined with a spherical-harmonic decomposition of the array.

Reference:
    D. P. Jarrett, E. A. P. Habets, M. R. P. Thomas and P. A. Naylor,
    "Rigid sphere room impulse response simulation: algorithm and
    applications", JASA 132(3), pp. 1462-1472, 2012.

The heavy inner loop runs in the native extension ``_smir_loop``; this module
mirrors the MATLAB wrapper (argument handling, mode-strength computation, FFT
reconstruction and the optional high-pass filter).
"""

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.signal import butter, lfilter

from . import _smir_loop
from .coords import sph2cart, cart2sph

__all__ = ["smir_generator", "smir_generator_batch", "order_per_freq"]


def order_per_freq(freq, c, sph_radius, factor=1.0, margin=0, n_min=0,
                   n_max=None):
    """Frequency-dependent spherical-harmonic order from the ``kr`` rule.

    Returns an integer order per frequency bin, ``N(f) = clip(ceil(factor *
    k*r) + margin, n_min, n_max)`` with ``k = 2*pi*f/c``. This matches the
    spatial bandwidth of a sphere of radius ``r`` (modes with ``n >> kr`` carry
    negligible energy), giving a more physical truncation and skipping useless
    high-order work at low frequencies.

    Pass the result (or this function via a lambda) as ``N_harm`` to
    :func:`smir_generator`, e.g.::

        N_harm=lambda f: order_per_freq(f, c, sphRadius, margin=2, n_max=30)

    Parameters
    ----------
    freq : array_like
        Frequency axis in Hz (one-sided), length ``K*nsample/2+1``.
    c : float
        Speed of sound (m/s).
    sph_radius : float
        Array radius (m).
    factor : float
        Multiplier on ``k*r`` (use >1 for a safety margin, default 1).
    margin : int
        Constant orders added on top.
    n_min, n_max : int or None
        Lower / upper clamp on the order (``n_max=None`` -> no upper clamp).
    """
    freq = np.asarray(freq, dtype=float)
    k = 2 * np.pi * freq / c
    n = np.ceil(factor * k * sph_radius).astype(int) + int(margin)
    n = np.maximum(n, int(n_min))
    if n_max is not None:
        n = np.minimum(n, int(n_max))
    return n


_MODE_STRENGTH_CACHE = {}
_HP_FILTER_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _mode_strength_cached(sph_type, sph_radius, k, n_harm):
    """Memoized wrapper around :func:`_mode_strength`.

    In dataset generation the mode strength is identical across every call that
    shares ``(sphType, sphRadius, k-grid, N_harm)`` (i.e. fixed array, sampling
    rate, nsample and order rule) yet the scipy spherical-Bessel evaluation
    dominates the runtime. Cache it on the byte content of the wavenumber grid
    so repeated calls reuse the result. Returns copies so callers may mutate the
    arrays freely. The cache is capped to avoid unbounded growth.

    Thread-safe: :func:`smir_generator_batch` calls this from worker threads, so
    the cache is guarded by a lock (taken only on the slow miss path).
    """
    key = (sph_type, float(sph_radius), int(n_harm), k.shape, k.tobytes())
    hit = _MODE_STRENGTH_CACHE.get(key)
    if hit is None:
        with _CACHE_LOCK:
            hit = _MODE_STRENGTH_CACHE.get(key)        # re-check inside the lock
            if hit is None:
                ms, n_eff = _mode_strength(sph_type, sph_radius, k, n_harm)
                if len(_MODE_STRENGTH_CACHE) > 32:
                    _MODE_STRENGTH_CACHE.clear()
                hit = (ms, n_eff)
                _MODE_STRENGTH_CACHE[key] = hit
    ms, n_eff = hit
    return ms.copy(), n_eff


def _mode_strength(sph_type, sph_radius, k, n_harm):
    """Far-field mode strength b_n(k r), shape (k_total, n_harm+1).

    Mirrors the rigid/open-sphere branches of ``smir_generator.m`` including
    the automatic reduction of ``n_harm`` when the Hankel recursion overflows.
    """
    kr = k * sph_radius                      # (k_total,)
    nonzero = kr > 0                         # DC bin (kr=0) is singular by design
    overflow_warned = False
    while True:
        orders = np.arange(n_harm + 1)
        # x[kk, l] = k[kk] * r ; evaluate per (order, freq)
        x = kr[:, None]                      # (k_total, 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if sph_type == "rigid":
                yn = spherical_yn(orders[None, :], x)
                djn = spherical_jn(orders[None, :], x, derivative=True)
                dyn = spherical_yn(orders[None, :], x, derivative=True)
                # Overflow guard (MATLAB besselh overflow workaround). The DC
                # bin is excluded because spherical_yn is -inf at the origin by
                # construction (it is later zeroed via the NaN path).
                if n_harm > 0 and np.any(np.isinf(yn[nonzero])):
                    n_harm -= 1
                    if not overflow_warned:
                        warnings.warn(
                            "N_harm too high; reducing until no overflow occurs.")
                        overflow_warned = True
                    continue
                hankel_deriv = djn + 1j * dyn               # d/dx h_n^(1)(x)
                ms = 1j / (hankel_deriv * (kr[:, None] ** 2))   # Wronskian
            else:  # open sphere
                ms = spherical_jn(orders[None, :], x)
        return ms, n_harm


def smir_generator(c, procFs, sphLocation, s, L, beta, sphType, sphRadius, mic,
                   N_harm, nsample=None, K=2, order=-1,
                   refl_coeff_ang_dep=0, HP=0, src_type="o", src_ang=None,
                   plot_geometry=False, fmin=0.0):
    """Generate spherical-array room impulse responses.

    Parameters
    ----------
    c : float
        Speed of sound (m/s).
    procFs : float
        Processing sampling frequency (Hz).
    sphLocation : (3,) array_like
        Array centre (x, y, z) in m.
    s : (3,) array_like
        Source position (x, y, z) in m.
    L : (3,) array_like
        Room dimensions (x, y, z) in m.
    beta : float or (6,) array_like
        If ``refl_coeff_ang_dep == 0``: either the reverberation time T60 (s,
        scalar) or the six wall reflection coefficients
        ``[x1 x2 y1 y2 z1 z2]``.  If ``refl_coeff_ang_dep == 1``: the six
        "effective" flow resistivities (typically 1e3 .. 1e9).
    sphType : {'open', 'rigid'}
        Microphone array type.
    sphRadius : float
        Array radius (m). ``0`` -> single microphone at the centre.
    mic : (M, 2) array_like
        Microphone angles ``(azimuth, inclination)`` in radians.
    N_harm : int, array_like, or callable
        Maximum spherical-harmonic order. Either a scalar (same order at every
        frequency), a per-bin array of length ``K*nsample/2+1`` giving the order
        for each frequency bin, or a callable mapping the frequency axis (Hz,
        one-sided) to such an array. Per-frequency orders let you match the
        order to the array's spatial bandwidth at each frequency (e.g.
        ``N(k) ~ k * sphRadius``), which is both more physical and faster.
        See :func:`smirgen.order_per_freq` for a ready-made rule.
    nsample : int, optional
        RIR length in samples. Defaults to ``T60 * procFs`` (only available
        when ``refl_coeff_ang_dep == 0``).
    K : int, optional
        Oversampling factor (default 2).
    order : int, optional
        Maximum reflection order (default -1 = unlimited).
    refl_coeff_ang_dep : {0, 1}, optional
        0 = real reflection coefficients, 1 = angle-dependent (Komatsu model).
    HP : {0, 1}, optional
        Apply a 4th-order 50 Hz high-pass filter.
    src_type : {'o', 's', 'c', 'h', 'b'}, optional
        Source directivity (omni / subcardioid / cardioid / hypercardioid /
        bidirectional).
    src_ang : (2,) array_like, optional
        Source look direction ``(azimuth, inclination)`` in radians. Defaults
        to pointing from the source towards the array centre.
    plot_geometry : bool, optional
        If True, draw the room/array/source geometry in 3D (requires
        matplotlib) and show it. For more control use
        :func:`smirgen.plot_geometry` directly.
    fmin : float, optional
        Lower frequency limit in Hz used **only** for the internal wavenumber
        grid (default 0 = disabled). Bins whose nominal frequency is below
        ``fmin`` are evaluated at ``fmin`` instead, which avoids the DC (0 Hz)
        singularity of the spherical mode strength (otherwise that bin is NaN
        and gets zeroed). The returned ``H``/``h`` keep their normal bin layout
        and length; only the *value* computed at those low bins changes.

    Returns
    -------
    h : (M, nsample) ndarray
        Room impulse responses.
    H : (M, K*nsample/2 + 1) ndarray (complex)
        Room transfer functions (one-sided).
    beta_hat : float or (6,) ndarray
        Reflection coefficient(s) derived from T60, or 0 when angle-dependent
        coefficients are used.
    """
    sphLocation = np.asarray(sphLocation, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    L = np.asarray(L, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    mic = np.asarray(mic, dtype=float)

    # ---- reflection coefficients / default RIR length --------------------
    if refl_coeff_ang_dep == 0:
        if beta.size == 1:
            V = L[0] * L[1] * L[2]
            S = 2 * (L[0] * L[2] + L[1] * L[2] + L[0] * L[1])
            TR = beta[0]
            if TR <= 0:
                raise ValueError("Scalar beta (reverberation time T60) must be > 0.")
            alfa = 24 * V * np.log(10) / (c * S * TR)   # Sabine-Franklin
            if alfa > 1:
                raise ValueError(
                    "The reflection coefficients cannot be calculated from the "
                    "supplied room dimensions and reverberation time. Supply the "
                    "reflection coefficients or change T60/room dimensions.")
            beta_hat = np.sqrt(1 - alfa)
            beta = np.full(6, beta_hat)
        else:
            beta_hat = beta.copy()

        if nsample is None:
            V = L[0] * L[1] * L[2]
            alpha = (((1 - beta[0] ** 2) + (1 - beta[1] ** 2)) * L[1] * L[2] +
                     ((1 - beta[2] ** 2) + (1 - beta[3] ** 2)) * L[0] * L[2] +
                     ((1 - beta[4] ** 2) + (1 - beta[5] ** 2)) * L[0] * L[1])
            TR = 24 * np.log(10.0) * V / (c * alpha)
            TR = max(TR, 0.128)
            nsample = int(np.ceil(TR * procFs))
    else:
        beta_hat = 0
        if nsample is None:
            raise ValueError(
                "nsample must be provided when refl_coeff_ang_dep == 1.")

    # ---- default look angle (towards the array centre) -------------------
    if src_ang is None:
        # NOTE: the original .m had a bug using (z-z) instead of the z
        # difference here; the z component is corrected below.
        az, inc, _ = cart2sph(sphLocation[0] - s[0],
                              sphLocation[1] - s[1],
                              sphLocation[2] - s[2])
        src_ang = np.array([az, inc], dtype=float)
    else:
        src_ang = np.asarray(src_ang, dtype=float).reshape(-1)

    # ---- sanity checks ----------------------------------------------------
    if sphLocation.size != 3:
        raise ValueError("sphLocation must be a length-3 vector.")
    if s.size != 3:
        raise ValueError("s must be a length-3 vector.")
    if L.size != 3:
        raise ValueError("L must be a length-3 vector.")
    if sphType not in ("open", "rigid"):
        raise ValueError("sphType must be 'open' or 'rigid'.")
    if mic.ndim != 2 or mic.shape[1] != 2:
        mic = mic.T if mic.ndim == 2 else mic.reshape(-1, 2)
        if mic.shape[1] != 2:
            raise ValueError("mic must be an (M, 2) matrix.")
    if refl_coeff_ang_dep not in (0, 1):
        raise ValueError("refl_coeff_ang_dep must be 0 or 1.")
    if beta.size not in (1, 6):
        raise ValueError("beta must be a scalar or a length-6 vector.")
    if beta.size == 1 and refl_coeff_ang_dep == 1:
        raise ValueError(
            "Angle dependent reflection coefficients must be a length-6 vector.")
    if src_type not in ("o", "s", "c", "h", "b"):
        raise ValueError("src_type must be one of 'o', 's', 'c', 'h', 'b'.")
    if src_ang.size != 2:
        raise ValueError("src_ang (look angle) must be a length-2 vector.")
    if np.linalg.norm(sphLocation - s) < sphRadius:
        warnings.warn(
            "The source cannot be inside the array. No impulse response computed.")
        N_FFT = K * nsample
        H = np.zeros((mic.shape[0], N_FFT // 2 + 1), dtype=complex)
        return np.zeros(0), H, 0
    if not (np.all(s <= L) and np.all(s >= 0)):
        raise ValueError("The source must be inside the room.")
    if not (np.all(sphLocation + sphRadius <= L) and
            np.all(sphLocation - sphRadius >= 0)):
        raise ValueError("The entire array must be inside the room.")

    # ---- look angle in (normalised) cartesian coordinates ----------------
    lx, ly, lz = sph2cart(src_ang[0], src_ang[1], 1.0)
    src_ang_cart = np.array([lx, ly, lz], dtype=float)
    src_ang_cart /= np.linalg.norm(src_ang_cart)

    if plot_geometry:
        from .plotting import plot_geometry as _plot_geometry
        # Build the figure without a blocking plt.show(): inside Jupyter an
        # interactive backend would otherwise stall here until the window is
        # closed. Display inline in notebooks, fall back to show() in scripts.
        fig, _ = _plot_geometry(L, sphLocation, s, sphRadius=sphRadius, mic=mic,
                                src_ang=src_ang, sphType=sphType, show=False)
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except Exception:
            in_notebook = False
        if in_notebook:
            from IPython.display import display
            display(fig)
        else:
            import matplotlib.pyplot as plt
            plt.show()

    N_FFT = K * nsample
    k_total = N_FFT // 2 + 1

    # Microphone unit-radius cartesian   positions (M x 3).
    mx, my, mz = sph2cart(mic[:, 0], mic[:, 1], 1.0)
    mic_pos = np.column_stack([mx, my, mz])

    # Frequency grid and wavenumbers. `fmin` clamps the frequencies used to
    # build the wavenumbers (so the 0 Hz / sub-fmin bins are evaluated at fmin),
    # while the output bin layout is unchanged.
    kk = np.arange(k_total)
    freq = kk * procFs / N_FFT
    freq_eff = np.maximum(freq, fmin) if fmin and fmin > 0 else freq
    with np.errstate(divide="ignore"):
        k = 2 * np.pi * freq_eff / c

    # ---- resolve N_harm (scalar / per-bin array / callable of freq) -------
    if callable(N_harm):
        n_per = np.asarray(N_harm(freq), dtype=int).reshape(-1)
    elif np.ndim(N_harm) > 0:
        n_per = np.asarray(N_harm, dtype=int).reshape(-1)
    else:
        n_per = None                                     # scalar -> uniform
    if n_per is not None:
        if n_per.size != k_total:
            raise ValueError(
                f"Per-frequency N_harm must have length k_total={k_total} "
                f"(K*nsample/2+1); got {n_per.size}.")
        if np.any(n_per < 0):
            raise ValueError("N_harm values must be non-negative.")
        N_max = int(n_per.max())
    else:
        N_max = int(N_harm)

    # ---- mode strength ----------------------------------------------------
    if sphRadius == 0:
        # Single microphone: mode strength is unused by the native loop.
        ms = np.zeros((k_total, N_max + 1), dtype=complex)
    else:
        ms, N_max = _mode_strength_cached(sphType, sphRadius, k, N_max)

    # The overflow guard may have lowered N_max; clip the per-bin orders to it.
    if n_per is not None:
        n_per = np.clip(n_per, 0, N_max)
    else:
        n_per = np.full(k_total, N_max, dtype=int)       # uniform order

    shd_k_l = 1j * ms * k[:, None]                       # (k_total, N_max+1)
    shd_angle_l = (2 * np.arange(N_max + 1) + 1).astype(float)

    # Native arrays must be column-major (Fortran) to match the loop indexing.
    shd_k_l = np.asfortranarray(shd_k_l.astype(np.complex128))
    mic_pos = np.asfortranarray(mic_pos.astype(np.float64))
    beta6 = np.ascontiguousarray(beta.astype(np.float64))
    n_per = np.ascontiguousarray(n_per.astype(np.int32))

    # ---- native inner loop ------------------------------------------------
    H = _smir_loop.smir_loop(
        float(c), float(procFs),
        np.ascontiguousarray(sphLocation), np.ascontiguousarray(s),
        np.ascontiguousarray(L), beta6,
        int(nsample), int(order), int(K),
        shd_k_l, shd_angle_l, mic_pos,
        float(sphRadius), np.ascontiguousarray(k.astype(np.float64)),
        int(refl_coeff_ang_dep), np.ascontiguousarray(src_ang_cart),
        src_type, n_per)

    H = np.asarray(H)
    H[np.isnan(H)] = 0

    # Conjugate to match MATLAB's exp(-ikR)/kR Fourier convention.
    H = np.conj(H)

    # One-sided -> real RIR (irfft rebuilds the Hermitian-symmetric spectrum the
    # way MATLAB's `ifft([H conj(H(:,N_FFT/2:-1:2))], ..., 'symmetric')` does),
    # then truncate the oversampled tail.
    h = np.fft.irfft(H, n=N_FFT, axis=1)
    h = h[:, :nsample]

    if HP == 1:
        ba = _HP_FILTER_CACHE.get(procFs)
        if ba is None:
            ba = butter(4, 50 / (procFs / 2), btype="high")
            _HP_FILTER_CACHE[procFs] = ba
        b, a = ba
        h = lfilter(b, a, h, axis=1)

    return h, H, beta_hat


def smir_generator_batch(varying, n_workers=None, return_H=False, **common):
    """Generate many RIRs in parallel — the fast path for building a dataset.

    Each RIR is an independent call to :func:`smir_generator`; the heavy
    image-source loop runs in the native extension with the GIL released, so a
    thread pool scales almost linearly with the number of cores (no pickling or
    array copying, and the mode-strength / high-pass caches are shared across
    workers). For a reverberant dataset (``order=-1``) this is typically ~7-8x
    on top of the per-call savings from a tight ``N_harm`` (see
    :func:`order_per_freq`).

    Parameters
    ----------
    varying : sequence of dict
        One dict per RIR, holding the keyword arguments that change between
        samples (e.g. ``s``, ``sphLocation``, ``L``, ``beta``). Each is merged
        on top of ``common`` to form the full :func:`smir_generator` call.
    n_workers : int, optional
        Thread-pool size. Defaults to ``os.cpu_count()`` (capped at the number
        of items). Past ~8 the speedup usually saturates on memory bandwidth.
    return_H : bool, optional
        If True, return the full ``(h, H, beta_hat)`` tuple per item; otherwise
        return just the RIR ``h`` (the common case). Default False.
    **common
        Keyword arguments shared by every call (``c``, ``procFs``, ``sphType``,
        ``sphRadius``, ``mic``, ``N_harm``, ``nsample``, ``K``, ``order`` ...).

    Returns
    -------
    list
        Results in the same order as ``varying``: either ``h`` arrays
        (``return_H=False``) or ``(h, H, beta_hat)`` tuples.

    Examples
    --------
    >>> freq = np.fft.rfftfreq(2048, 1 / 16000)
    >>> N_harm = order_per_freq(freq, c=343, sph_radius=0.105, margin=4, n_max=30)
    >>> varying = [dict(sphLocation=ctr, s=src, L=room, beta=t60)
    ...            for ctr, src, room, t60 in samples]
    >>> rirs = smir_generator_batch(
    ...     varying, c=343, procFs=16000, sphType="rigid", sphRadius=0.105,
    ...     mic=mic, N_harm=N_harm, nsample=2048, K=1, order=-1, HP=1, fmin=10)
    """
    items = list(varying)
    if not items:
        return []

    def _run(params):
        out = smir_generator(**{**common, **params})
        return out if return_H else out[0]

    # Warm the shared caches single-threaded on the first item so the worker
    # threads only ever read them (the mode strength is identical across the
    # batch when the array / nsample / N_harm are fixed in ``common``).
    first = _run(items[0])
    if len(items) == 1:
        return [first]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(items) - 1))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rest = list(ex.map(_run, items[1:]))
    return [first, *rest]
