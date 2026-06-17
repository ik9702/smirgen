# Changelog

## 2.7.1
- **`SmirArray.generate(..., return_H=True)`**: optionally also return the
  one-sided complex transfer function ``H`` of shape ``(N, M, K*nsample/2+1)``,
  matching :func:`smir_generator`'s ``H`` output (to ~1e-8 in complex128). The
  RIR ``h`` is still returned first; ``return_H`` is off by default.

## 2.7.0
- **GPU backend `SmirArray` (PyTorch)**: fast-path dataset generation on the GPU
  for the common case where the room, array centre, microphones and radius are
  **fixed** and only the **source position** varies. Covers spherical arrays
  (open/rigid), omnidirectional sources and real reflection coefficients.
  Source-independent state (mode strength, wavenumber grid, image-source lattice
  and reflection factors) is precomputed once on the host; per-source work — the
  Hankel/Legendre expansion summed over images — runs as batched matmuls plus a
  batched inverse FFT. Construct once, then ``arr.generate(sources)`` returns
  ``(N, M, nsample)``.
  - Image lattice is exactly pruned at construction to the cells reachable by
    some in-room source (typically ~2.5x fewer candidates).
  - Output matches `smir_generator` to ~1e-7 in `complex128` and ~1.5e-5 in the
    default `complex64` (faster / half the memory on GPU). The 50 Hz high-pass
    is always applied in float64 (its poles sit at z≈1, so float32 is unstable).
  - Requires PyTorch (`pip install smirgen[gpu]`); the import is lazy so the
    rest of the package works without torch.

## 2.6.0
- **`smir_generator_batch`**: generate many RIRs in parallel for dataset
  building. Each RIR is an independent `smir_generator` call whose native
  image-source loop releases the GIL, so a thread pool scales almost linearly
  with cores (no pickling/copying; the mode-strength and high-pass caches are
  shared and warmed once up front). For a reverberant dataset (`order=-1`) this
  is ~7-8x on top of the per-call savings from a tight `N_harm`. Pass the fixed
  array/room config as keyword arguments and a list of per-sample overrides;
  results come back in input order. The cache is now lock-guarded for safe
  concurrent use.

## 2.5.1
- **Performance (dataset generation)**: cache the spherical mode strength and
  the high-pass filter coefficients across calls. In a generation loop with a
  fixed array, sampling rate, `nsample` and `N_harm` rule these are identical
  every call, yet the scipy spherical-Bessel evaluation dominated the runtime
  (~77% of a call). Repeated calls that only vary the source/room geometry now
  reuse the cached result. Output unchanged to machine precision; ~10x faster
  per call in typical dataset loops.

## 2.5.0
- **Frequency-dependent `N_harm`**: `smir_generator` now accepts a scalar, a
  per-frequency-bin array, or a callable `freq -> order`. The native loop sums
  only up to the per-bin order (faster and more physical). Added
  `order_per_freq` helper implementing the `N ~ k*r` rule.
- **`fmin` option**: clamp the internal wavenumber grid to a minimum frequency
  so the 0 Hz / sub-`fmin` bins are evaluated at `fmin` (avoids the DC
  singularity), without changing the output bin layout.
- **`relative_db` / `direct_reference`**: express `|H|` in dB relative to the
  source (1 m free field, `ref="source"`) or to the direct sound
  (`ref="direct"`).
- **3D geometry plot**: `plot_geometry()` and `smir_generator(...,
  plot_geometry=True)` (non-blocking, notebook-friendly), with an array
  close-up panel.

## 2.2.0
- Performance: hoisted the reflection/directivity factor out of the inner
  `(mic x frequency)` loops in the native core. Output unchanged to machine
  precision; up to ~18x faster for angle-dependent reflection coefficients.

## 2.0.0
- Initial Python port of the SMIR generator: NumPy/SciPy wrapper around a
  pybind11 port of the original MATLAB MEX core (`smir_generator_loop.cpp`).
  Fixes carried over from the review are documented in `CODE_REVIEW.md`.
