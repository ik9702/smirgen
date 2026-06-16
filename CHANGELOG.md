# Changelog

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
