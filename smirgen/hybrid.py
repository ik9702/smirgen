"""Hybrid ISM + statistical-tail RIR generation for spherical arrays.

The pure image-source method (:func:`smir_generator`) places every reflection
exactly, but the number of image sources grows with the cube of the RIR length
(``~ T60**3 * c**3 / V``).  For a large, reverberant room a long, fully
geometric RIR is therefore expensive — which is the bottleneck even on the GPU
backend.

For machine-learning datasets the *late* reverberation does not need to be
geometrically exact: by the mixing time the reflection density is high enough
that the field is, to a good approximation, **diffuse** (spatially isotropic
and exponentially decaying).  This module keeps the cheap, exact early part
from the image-source method and replaces the expensive late tail with a fast
statistical model:

1.  **Early part** — :func:`smir_generator` with a bounded reflection ``order``
    (only the strong, low-order reflections; cheap).
2.  **Late tail** — a multichannel diffuse-noise field whose *inter-microphone
    coherence* matches the array's diffuse-field coherence (open sphere:
    ``sinc(k d)``; rigid sphere: the mode-strength sum), shaped by an
    exponential energy decay set by the room's T60.  The two are crossfaded at
    the mixing time.

The output is **not** sample-identical to a full image-source RIR: the tail is
a statistically-matched realisation, equal to the geometric tail only in its
energy-decay (T60) and spatial-coherence statistics.  That is the right
trade-off for dereverberation / source-separation training data, where those
statistics — not the exact late echo pattern — are what matters.

The diffuse coherence-constrained noise follows
    E. A. P. Habets, I. Cohen, S. Gannot, "Generating nonstationary multisensor
    signals under a spatial coherence constraint", JASA 124(5), 2008,
applied here per rFFT bin (the field is stationary; the common exponential
envelope is a real per-sample gain and so leaves the inter-channel coherence
unchanged).
"""

import numpy as np

from .generator import smir_generator, _mode_strength_cached, order_per_freq
from .coords import sph2cart

__all__ = ["smir_generator_hybrid", "hybrid_params"]


def _mixing_time(V):
    """Perceptual mixing time (s) from room volume — ``~2*sqrt(V)`` ms.

    The point past which the field is well-approximated as diffuse; the
    statistical tail takes over here.  Follows the volume rule of Lindau,
    Kosanke & Weinzierl, "Perceptual evaluation of model- and signal-based
    predictors of the mixing time in binaural room impulse responses" (2012).
    """
    return 2.0 * np.sqrt(float(V)) * 1e-3


def _auto_early_order(L, c, procFs, mix_time, margin=0):
    """Reflection order whose images fill all arrivals up to ``mix_time``.

    The image method's ``order`` bounds the reflection order, not time; an
    order-``p`` image can arrive as early as ``p * min(L) / c``.  To make the
    geometric early part dense up to the mixing time we therefore need an order
    of at least ``c * mix_time / min(L)`` (the most reflections that can pack
    into ``mix_time`` bouncing off the shortest dimension).  This mirrors
    pyroomacoustics' ``inverse_sabine`` order rule (``c * rt60 / min(L)``) but
    with the *mixing time* in place of the full T60 — the whole point of the
    hybrid is that the image method only has to reach the mixing time.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    return max(2, int(np.ceil(c * mix_time / np.min(L))) + int(margin))


def _t60_from_room(L, beta, c):
    """Eyring/Sabine T60 (s) from room dims and the six wall coefficients.

    ``beta`` is a length-6 array of pressure reflection coefficients
    ``[x1 x2 y1 y2 z1 z2]`` (so wall absorption is ``alpha = 1 - beta**2``).
    Uses the same Sabine-Franklin constant as :func:`smir_generator`'s default
    ``nsample`` rule, so the decay matches the geometric model's expected
    length.
    """
    L = np.asarray(L, dtype=float)
    beta = np.asarray(beta, dtype=float)
    V = L[0] * L[1] * L[2]
    # Per-wall absorbed-area = (1 - beta**2) * wall_area, summed over 6 walls.
    a = (((1 - beta[0] ** 2) + (1 - beta[1] ** 2)) * L[1] * L[2] +
         ((1 - beta[2] ** 2) + (1 - beta[3] ** 2)) * L[0] * L[2] +
         ((1 - beta[4] ** 2) + (1 - beta[5] ** 2)) * L[0] * L[1])
    return 24 * np.log(10.0) * V / (c * a)


def hybrid_params(L, rt60, sphRadius, c=343.0, procFs=16000.0, sphType="rigid",
                  K=1, mix_time=None, order_margin=0, n_harm_margin=2,
                  n_max=30, fmin=0.0):
    """Pick sensible :func:`smir_generator_hybrid` parameters from room + T60.

    The companion to pyroomacoustics' ``inverse_sabine``: given the room
    dimensions and the desired reverberation time, it returns a ready-to-use
    keyword dict (reflection coefficient, RIR length, mixing time, early
    reflection order and a frequency-matched harmonic order).  You still supply
    the geometry that varies per RIR — ``sphLocation``, ``s`` and ``mic`` — when
    you make the call::

        kw = hybrid_params(L=[10, 8, 4], rt60=0.8, sphRadius=0.042)
        h, beta_hat = smir_generator_hybrid(
            sphLocation=ctr, s=src, mic=mic, **kw)

    Parameters
    ----------
    L : (3,) array_like
        Room dimensions (m).
    rt60 : float
        Target reverberation time T60 (s).
    sphRadius : float
        Array radius (m) — sets the harmonic-order rule and the diffuse
        coherence.
    c, procFs, sphType, K, fmin :
        As in :func:`smir_generator_hybrid`.
    mix_time : float, optional
        Override the mixing time (s).  Default ``~2*sqrt(V)`` ms.
    order_margin : int, optional
        Added to the auto early reflection order (raise for a longer, more
        accurate geometric early part).
    n_harm_margin, n_max : int, optional
        Margin and cap for the per-frequency harmonic order
        (:func:`order_per_freq`).

    Returns
    -------
    params : dict
        Keyword arguments for :func:`smir_generator_hybrid`: ``c``, ``procFs``,
        ``L``, ``beta``, ``sphType``, ``sphRadius``, ``N_harm``, ``nsample``,
        ``K``, ``early_order``, ``mix_time``, ``fmin``.  Also carries an
        ``info`` key (popped before use, or ignored — it is accepted and dropped
        by the generator) describing the derived quantities: ``beta_hat`` (the
        reflection coefficient), ``mix_time_ms``, ``early_order`` and
        ``full_order`` (the order a *full* geometric RIR would need, for
        comparison — typically far larger).
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    V = float(L[0] * L[1] * L[2])
    S = 2 * (L[0] * L[1] + L[1] * L[2] + L[0] * L[2])
    if rt60 <= 0:
        raise ValueError("rt60 must be > 0.")
    alpha = 24 * np.log(10.0) * V / (c * S * rt60)           # Sabine absorption
    if alpha >= 1:
        raise ValueError(
            f"rt60={rt60}s is too short for room {tuple(L)} (Sabine alpha="
            f"{alpha:.2f} >= 1). Increase rt60 or the room size.")
    beta_hat = float(np.sqrt(1 - alpha))

    if mix_time is None:
        mix_time = _mixing_time(V)
    early_order = _auto_early_order(L, c, procFs, mix_time, margin=order_margin)
    full_order = max(1, int(np.ceil(c * rt60 / np.min(L))))  # pyroomacoustics rule

    nsample = int(np.ceil(max(rt60, 0.128) * procFs))
    N_FFT = K * nsample
    freq = np.arange(N_FFT // 2 + 1) * procFs / N_FFT
    N_harm = order_per_freq(freq, c=c, sph_radius=sphRadius,
                            margin=n_harm_margin, n_max=n_max)

    return {
        "c": c, "procFs": procFs, "L": L, "beta": float(rt60),
        "sphType": sphType, "sphRadius": sphRadius, "N_harm": N_harm,
        "nsample": nsample, "K": K, "early_order": early_order,
        "mix_time": float(mix_time), "fmin": fmin,
        "info": {"beta_hat": beta_hat, "mix_time_ms": mix_time * 1e3,
                 "early_order": early_order, "full_order": full_order,
                 "nsample": nsample},
    }


def _diffuse_coherence(sphType, sphRadius, mic_pos, k, n_harm):
    """Spatial coherence matrix of a spherically-isotropic diffuse field.

    Returns ``Gamma`` of shape ``(k_total, M, M)`` (real, symmetric, unit
    diagonal): the expected normalised cross-spectrum between every pair of
    microphones in a diffuse field, per wavenumber bin ``k``.

    - **open** sphere (pressure on the sphere): the classic 3-D result
      ``Gamma_mn = sinc(k * d_mn)`` with ``d_mn`` the chord distance between
      mics ``m`` and ``n``.
    - **rigid** sphere: scattering modifies it; using the addition theorem the
      diffuse coherence is the mode-strength-weighted Legendre sum
      ``Gamma_mn = sum_n (2n+1) |b_n(kr)|^2 P_n(cos g_mn) /
                   sum_n (2n+1) |b_n(kr)|^2`` with ``g_mn`` the angle between
      the two microphone directions and ``b_n`` the array mode strength.
    """
    M = mic_pos.shape[0]
    k = np.asarray(k, dtype=float)
    k_total = k.shape[0]

    # Unit microphone directions (mic_pos rows are already unit vectors).
    cos_g = np.clip(mic_pos @ mic_pos.T, -1.0, 1.0)            # (M, M)

    if sphType == "open" or sphRadius == 0.0:
        # Chord distance d = r * |u_m - u_n| = r * sqrt(2 - 2 cos g).
        d = sphRadius * np.sqrt(np.maximum(2.0 - 2.0 * cos_g, 0.0))   # (M, M)
        x = k[:, None, None] * d[None, :, :]                  # (k_total, M, M)
        gamma = np.sinc(x / np.pi)                            # sin(x)/x
        return gamma

    # ---- rigid sphere: mode-strength-weighted Legendre sum ----------------
    ms, n_eff = _mode_strength_cached(sphType, sphRadius, k, n_harm)   # (k_total, n+1)
    w = (2 * np.arange(n_eff + 1) + 1)[None, :] * np.abs(ms) ** 2      # (k_total, n+1)

    # Legendre polynomials P_n(cos g) for every order and mic pair (recursion).
    P_prev = np.ones_like(cos_g)                              # P_0
    num = w[:, 0][:, None, None] * P_prev[None, :, :]
    if n_eff >= 1:
        P_cur = cos_g                                         # P_1
        num = num + w[:, 1][:, None, None] * P_cur[None, :, :]
        for n in range(2, n_eff + 1):
            P_next = ((2 * n - 1) * cos_g * P_cur - (n - 1) * P_prev) / n
            num = num + w[:, n][:, None, None] * P_next[None, :, :]
            P_prev, P_cur = P_cur, P_next

    denom = w.sum(axis=1)                                     # (k_total,)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = num / denom[:, None, None]
    gamma[~np.isfinite(gamma)] = 0.0
    # DC / kr=0 bin: perfectly coherent (uniform pressure).
    gamma[denom == 0] = 1.0
    return gamma


def _coherent_diffuse_noise(gamma, n_samples, rng):
    """Multichannel white noise with inter-channel coherence ``gamma(f)``.

    ``gamma`` is ``(n_freq, M, M)`` on the one-sided rFFT grid of length
    ``n_samples`` (``n_freq = n_samples // 2 + 1``).  Returns ``(M, n_samples)``
    real noise: each channel is temporally white (flat PSD), and every channel
    pair has the prescribed cross-coherence.  Per-bin mixing via the
    eigendecomposition ``gamma = U diag(L) U^H`` -> ``C = U sqrt(L)``.
    """
    n_freq, M, _ = gamma.shape
    # Symmetric eigendecomposition per frequency bin (batched).
    evals, evecs = np.linalg.eigh(gamma)                     # (n_freq, M), (n_freq, M, M)
    evals = np.clip(evals, 0.0, None)
    C = evecs * np.sqrt(evals)[:, None, :]                   # (n_freq, M, M)

    # Independent unit-variance complex spectra, one per channel.
    white = (rng.standard_normal((n_freq, M)) +
             1j * rng.standard_normal((n_freq, M))) / np.sqrt(2.0)
    mixed = np.einsum("fij,fj->fi", C, white)               # (n_freq, M)

    x = np.fft.irfft(mixed.T, n=n_samples, axis=1)          # (M, n_samples)
    # Normalise to unit per-sample power (eigh mixing preserves coherence, the
    # absolute level is set later by the seam match).
    rms = np.sqrt(np.mean(x ** 2)) or 1.0
    return x / rms


def smir_generator_hybrid(
        c, procFs, sphLocation, s, L, beta, sphType, sphRadius, mic, N_harm,
        nsample=None, K=2, early_order=None, mix_time=None, HP=0, src_type="o",
        src_ang=None, fmin=0.0, seed=None, tail_gain=1.0, xfade_ms=2.0,
        return_parts=False, info=None):
    """RIR with an exact early part and a fast statistical diffuse tail.

    Drop-in companion to :func:`smir_generator` for long RIRs in large rooms:
    the early reflections are computed exactly by the image-source method (up to
    ``early_order``), and everything past the mixing time is replaced by a
    diffuse-field noise tail with the correct spatial coherence and T60 decay.
    See the module docstring for the model and its limitations.

    Parameters
    ----------
    c, procFs, sphLocation, s, L, beta, sphType, sphRadius, mic, N_harm, K,
    HP, src_type, src_ang, fmin :
        As in :func:`smir_generator`.  ``beta`` must be a scalar T60 (s) or six
        real reflection coefficients (``refl_coeff_ang_dep == 1`` is not
        supported — the diffuse tail assumes real, frequency-flat walls).
    nsample : int, optional
        Total RIR length in samples.  Defaults to the T60-based length, exactly
        as :func:`smir_generator`.
    early_order : int, optional
        Maximum reflection order computed exactly by the image method.  Larger
        -> more of the RIR is geometric (slower, more accurate); smaller -> the
        statistical tail starts sooner (faster).  Default ``None`` derives it
        from the mixing time and room size (see :func:`hybrid_params`).
    info : dict, optional
        Ignored — accepted so the dict from :func:`hybrid_params` can be passed
        with ``**params`` directly.
    mix_time : float, optional
        Crossover (mixing) time in seconds.  Defaults to the perceptual mixing
        time ``~ 2 * sqrt(V) ms`` (Lindau/Polack), clamped into the RIR.
    seed : int, optional
        Seed for the diffuse-tail RNG (reproducible datasets).
    tail_gain : float, optional
        Multiplier on the matched tail level (default 1).  Use to calibrate the
        seam if the bounded-order early part under/over-estimates the energy at
        the mixing time.
    xfade_ms : float, optional
        Raised-cosine crossfade length (ms) around the mixing time.
    return_parts : bool, optional
        If True, also return a dict with the early RIR, the tail, the mixing
        sample and the T60 used (for inspection / tuning).

    Returns
    -------
    h : (M, nsample) ndarray
        The hybrid room impulse responses.
    beta_hat : float or (6,) ndarray
        Reflection coefficient(s), as :func:`smir_generator` returns.
    parts : dict, optional
        Present only if ``return_parts``.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    mic = np.asarray(mic, dtype=float)

    # ---- reflection coefficients, T60 and default length ------------------
    if beta.size == 1:
        V = L[0] * L[1] * L[2]
        S = 2 * (L[0] * L[2] + L[1] * L[2] + L[0] * L[1])
        T60 = float(beta[0])
        if T60 <= 0:
            raise ValueError("Scalar beta (T60) must be > 0.")
        alfa = 24 * V * np.log(10) / (c * S * T60)            # Sabine-Franklin
        if alfa > 1:
            raise ValueError(
                "Reflection coefficients cannot be derived from these room "
                "dimensions and T60; supply 6 coefficients or change T60/L.")
        beta_hat = np.sqrt(1 - alfa)
        beta6 = np.full(6, beta_hat)
    elif beta.size == 6:
        beta6 = beta.copy()
        beta_hat = beta.copy()
        T60 = _t60_from_room(L, beta6, c)
    else:
        raise ValueError("beta must be a scalar T60 or 6 reflection coefficients.")

    if nsample is None:
        nsample = int(np.ceil(max(T60, 0.128) * procFs))

    # ---- mixing time / crossover sample -----------------------------------
    V = L[0] * L[1] * L[2]
    if mix_time is None:
        mix_time = _mixing_time(V)                            # ~2*sqrt(V) ms
    n_mix = int(np.clip(round(mix_time * procFs), 1, nsample - 1))

    # ---- early reflection order (auto: cover arrivals up to the mixing time)
    if early_order is None:
        early_order = _auto_early_order(L, c, procFs, mix_time)

    # ---- exact early part (bounded reflection order, no HP yet) -----------
    h_early, _, _ = smir_generator(
        c=c, procFs=procFs, sphLocation=sphLocation, s=s, L=L, beta=beta6,
        sphType=sphType, sphRadius=sphRadius, mic=mic, N_harm=N_harm,
        nsample=nsample, K=K, order=int(early_order), refl_coeff_ang_dep=0,
        HP=0, src_type=src_type, src_ang=src_ang, fmin=fmin)
    M = h_early.shape[0]

    # ---- diffuse tail -----------------------------------------------------
    n_tail = nsample - n_mix
    rng = np.random.default_rng(seed)

    # Microphone unit directions and the array diffuse-field coherence on the
    # tail's own rFFT grid.
    mxv, myv, mzv = sph2cart(mic[:, 0], mic[:, 1], 1.0)
    mic_pos = np.column_stack([mxv, myv, mzv])
    n_freq = n_tail // 2 + 1
    fk = np.fft.rfftfreq(n_tail, 1.0 / procFs)
    fk_eff = np.maximum(fk, fmin) if fmin and fmin > 0 else fk
    k_tail = 2 * np.pi * fk_eff / c
    n_harm_tail = int(N_harm(fk) .max()) if callable(N_harm) else (
        int(np.max(N_harm)) if np.ndim(N_harm) > 0 else int(N_harm))

    if M == 1 or sphRadius == 0.0:
        tail = rng.standard_normal((M, n_tail))
        tail /= np.sqrt(np.mean(tail ** 2)) or 1.0
    else:
        gamma = _diffuse_coherence(sphType, sphRadius, mic_pos, k_tail, n_harm_tail)
        tail = _coherent_diffuse_noise(gamma, n_tail, rng)

    # Exponential energy decay e(t) = 10^(-6 t / T60); amplitude = sqrt(e).
    t = np.arange(n_tail) / procFs
    env = np.power(10.0, -3.0 * t / T60)                      # sqrt of 10^(-6 t/T60)
    tail = tail * env[None, :]

    # ---- seam level: match the early RIR's RMS just before the mixing time
    w = max(1, int(0.005 * procFs))                          # 5 ms window
    seam_rms = np.sqrt(np.mean(h_early[:, max(0, n_mix - w):n_mix] ** 2))
    tail0_rms = np.sqrt(np.mean(tail[:, :w] ** 2)) or 1.0
    tail *= tail_gain * (seam_rms / tail0_rms)

    # ---- crossfade the early tail out / the diffuse tail in ---------------
    h = h_early.copy()
    nx = max(1, int(xfade_ms * 1e-3 * procFs))
    nx = min(nx, n_tail, n_mix)
    fade = 0.5 * (1 - np.cos(np.pi * np.arange(nx) / nx))     # 0->1 raised cosine
    # Fade the geometric tail down across [n_mix-nx, n_mix) ...
    h[:, n_mix - nx:n_mix] *= (1 - fade)[None, :]
    h[:, n_mix:] = 0.0
    # ... and add the diffuse tail, faded up over its first nx samples.
    tail_faded = tail.copy()
    tail_faded[:, :nx] *= fade[None, :]
    h[:, n_mix:] += tail_faded

    # ---- optional 50 Hz high-pass on the combined RIR ---------------------
    if HP == 1:
        from scipy.signal import butter, lfilter
        from .generator import _HP_FILTER_CACHE
        ba = _HP_FILTER_CACHE.get(procFs)
        if ba is None:
            ba = butter(4, 50 / (procFs / 2), btype="high")
            _HP_FILTER_CACHE[procFs] = ba
        b, a = ba
        h = lfilter(b, a, h, axis=1)

    if return_parts:
        return h, beta_hat, {
            "h_early": h_early, "tail": tail, "n_mix": n_mix, "T60": T60,
            "mix_time": n_mix / procFs}
    return h, beta_hat
