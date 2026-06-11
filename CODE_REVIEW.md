# Code review — SMIR generator (original MATLAB/MEX)

Review of `../smir_generator.m`, `../smir_generator_loop.cpp`, and the helper
files. Findings are grouped by severity. The Python port in this folder fixes
the items marked **[fixed in port]**.

## Bugs

1. **Wrong z-component in the default look angle** — `smir_generator.m:163`
   ```matlab
   [src_ang(1),src_ang(2)] = mycart2sph(sphLocation(1)-s(1), sphLocation(2)-s(2), sphLocation(2)-s(2));
   ```
   The third argument repeats `sphLocation(2)-s(2)` instead of
   `sphLocation(3)-s(3)`. The default source look-direction (used only when
   `src_ang` is omitted) is therefore computed with a wrong z component. The
   `run_*` example scripts pass `src_ang` explicitly and so never hit it, which
   is why it went unnoticed. **[fixed in port]** (`generator.py` uses the z
   difference).

2. **Typo'd variable name in a sanity check** — `smir_generator.m:193`
   ```matlab
   if (length(beta) == 1 && refl_coef_ang_dep == 1)   % missing the second 'f'
   ```
   The variable is `refl_coeff_ang_dep`. Because of `&&` short-circuiting this
   only evaluates when `beta` is scalar — in which case MATLAB throws
   *"undefined variable refl_coef_ang_dep"* instead of the intended clear error
   message. **[fixed in port]** (correct name, proper `ValueError`).

3. **`sphbesselh` / `legendre` write out of bounds for `N_harm == 0`** —
   `smir_generator_loop.cpp:351-352, 361-362`. Both routines unconditionally
   write `output[1]`, but the caller sizes the buffer to `N_harm+1`. With
   `N_harm = 0` (legal for a single sensor / low-order use) this is a one-element
   overflow → heap corruption. In the original this is masked because the
   auto-reduction never reaches 0; in the port the same path was reachable, so
   both routines now early-return when `max_nu < 1`. **[fixed in port]**.

## Fragile / sharp edges

4. **`size(...) ~= [1,2]` comparisons** — `smir_generator.m:212-214`. Comparing a
   size vector to `[1,2]` yields a logical array; `if` then requires *all*
   elements true, so the branch logic is accidental rather than intended. Works
   for the common shapes but is brittle. The port validates `.size`/`.shape`
   explicitly. **[fixed in port]**.

5. **DC-bin overflow detection** — `smir_generator.m:248`. The overflow guard
   tests `isinf(besselh(...))` over the *whole* matrix, including the `k = 0`
   (DC) column where the spherical Hankel function is singular by definition.
   MATLAB happens to return `NaN` (not `Inf`) there, so the guard survives; a
   naive SciPy reproduction returns `-inf` and would wrongly drive `N_harm` to 0
   (and then trip bug #3). The port excludes the DC bin from the check and
   relies on the existing `H(isnan)=0` path to zero that bin. **[fixed in port]**.

6. **`legendre` uses single-precision recurrence coefficients** —
   `smir_generator_loop.cpp:365`: `(float)(2*n-1)/n`. Casting the rational
   coefficients to `float` injects ~1e-7 relative error into an otherwise
   double-precision Legendre recursion. Kept **as-is** in the port for
   bit-compatibility with the original, but it is a gratuitous precision loss.

## Correctness notes (not bugs, but worth knowing)

7. **`mycart2sph` returns inclination, not elevation.** The docstring says
   "elevation … from xy plane", but the body computes `elev = acos(z./r)`, which
   is the **inclination from +z**. This is internally consistent with
   `mysph2cart` (which also uses inclination), so results are correct — only the
   documentation is misleading. The port documents the physics convention
   clearly.

8. **`N_FFT = K*nsample` is assumed even.** `H` is built as the one-sided
   spectrum of length `N_FFT/2+1` and `ifft([H conj(H(:,N_FFT/2:-1:2))])`
   assumes an even length. Odd `K*nsample` would misbehave. Unchanged in the
   port (`irfft(..., n=N_FFT)` carries the same assumption).

9. **Conjugation convention.** `H = conj(H)` before the inverse FFT is
   deliberate — it converts Williams' `exp(+ikR)` convention to MATLAB's
   `exp(-ikR)/kR`. The port reproduces this exactly; note `numpy.fft.irfft`
   already rebuilds the Hermitian-symmetric spectrum, so it is applied **once**
   (a double-conjugation bug was caught and removed during the port).

## Design / performance observations

- The image-source loop recomputed the reflection product and source
  directivity (and, for angle-dependent coefficients, `refl_factor_komatsu`)
  inside the innermost `(ang, kk)` loop, even though they depend only on the
  image (and, for Komatsu, on `kk`) — not on the microphone `ang`. **Fixed in
  the port:** these are now hoisted to once per image (or once per frequency
  bin), and the constant factor is pulled out of the spherical-harmonic `ll`
  sum. Output is unchanged to machine precision (max abs diff ~1e-15);
  measured speedups range from ~1.9x (real coefficients, 32 mics) to ~18x
  (angle-dependent coefficients, 32 mics).
- `pow(Q[i], abs(...))` with integer exponents goes through `std::pow` on
  complex doubles every iteration; integer power-by-squaring would be faster.
- The GIL is released around the loop in the port (`gil_scoped_release`), so
  multiple arrays/rooms can be generated concurrently from Python threads.
