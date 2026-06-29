# smirgen — Spherical Microphone array Impulse Response generator (Python)

`smirgen` generates room impulse responses (RIRs) between a sound source and an
**open or rigid spherical microphone array** in a reverberant shoebox room,
using the image-source method combined with a spherical-harmonic decomposition
of the array.

It is a Python port of the **SMIR Generator** (MATLAB + MEX) from the
International Audio Laboratories Erlangen. The numerically heavy image-source
loop is kept in native C++ (via **pybind11**); the rest is NumPy/SciPy, so
**no MATLAB is required**.

> Jarrett, Habets, Thomas & Naylor, *"Rigid sphere room impulse response
> simulation: algorithm and applications"*, JASA 132(3), pp. 1462–1472, 2012.

## Features

- Open / rigid spherical arrays, arbitrary microphone layouts, single sensor.
- Reverberation from T60 or per-wall reflection coefficients; angle-dependent
  (Komatsu) coefficients; reflection-order limit.
- Directional sources (omni / sub-/cardioid / hyper / bidirectional).
- **Frequency-dependent spherical-harmonic order** (`N_harm` as scalar, per-bin
  array, or callable) — more physical and faster.
- **`fmin`** option to avoid the DC singularity.
- **Fast dataset generation**: multi-threaded CPU batches (`smir_generator_batch`)
  and a **GPU (PyTorch) backend** (`SmirArray`) for the fixed-array /
  varying-source workload — see [Fast dataset generation](#fast-dataset-generation).
- **Hybrid generation for long RIRs in large rooms** (`smir_generator_hybrid`):
  exact early reflections + a fast diffuse-field statistical tail, ~100–300×
  faster for long, reverberant RIRs — see [Long RIRs](#long-rirs--hybrid-generation).
- Source/direct-referenced level analysis (`relative_db`).
- 3D geometry visualisation (`plot_geometry`), notebook-friendly.

## Install

Requires Python ≥ 3.8, NumPy, SciPy and a C++ compiler (g++/clang). `pybind11`
is pulled in automatically at build time.

```bash
pip install .                 # from the repository root
# editable install for development:
pip install -e .
# optional extras:
pip install ".[plot,test]"    # matplotlib, pytest
pip install ".[gpu]"          # PyTorch, for the SmirArray GPU backend
```

Verify:

```bash
python examples/example.py
pytest                        # if installed with [test]
```

## Usage

```python
import numpy as np
from smirgen import smir_generator

h, H, beta_hat = smir_generator(
    c=343,                        # speed of sound (m/s)
    procFs=8000,                  # sampling frequency (Hz)
    sphLocation=[1.6, 4.05, 1.7], # array centre (x,y,z) m
    s=[3.37, 4.05, 1.7],          # source (x,y,z) m
    L=[5, 6, 4],                  # room dims (x,y,z) m
    beta=0.3,                     # T60 (scalar) OR 6 reflection coeffs OR 6 flow resistivities
    sphType="rigid",             # "open" | "rigid"
    sphRadius=0.042,              # array radius (m); 0 -> single mic at centre
    mic=[[np.pi/4, np.pi/2],      # (azimuth, inclination) per mic, radians
         [np.pi/2, np.pi/2]],
    N_harm=30,                    # max spherical-harmonic order (see below)
    nsample=2048,                 # RIR length (default: T60*procFs)
    K=1,                          # oversampling factor (default 2)
    order=6,                      # max reflection order (-1 = unlimited)
    refl_coeff_ang_dep=0,         # 0 = real coeffs, 1 = angle-dependent (Komatsu)
    HP=1,                         # 50 Hz high-pass filter on the RIR
    src_type="o",                # o/s/c/h/b directivity
    src_ang=None,                 # (az, inc) look angle; default points at array
    fmin=0.0,                     # clamp internal freq grid to >= fmin Hz (0 = off)
    plot_geometry=False,          # draw the 3D setup
)
# h: (M, nsample) RIRs   H: (M, K*nsample/2+1) transfer functions   beta_hat: derived coeff
```

`beta` semantics (same as the MATLAB version):

| `refl_coeff_ang_dep` | `beta` scalar              | `beta` length-6 |
|----------------------|----------------------------|-----------------|
| `0`                  | reverberation time T60 (s) | wall reflection coefficients `[x1 x2 y1 y2 z1 z2]` |
| `1`                  | *(not allowed)*            | effective flow resistivities (≈1e3…1e9) |

### Frequency-dependent harmonic order

`N_harm` may be a scalar, a per-bin array of length `K*nsample/2+1`, or a
callable mapping the frequency axis to such an array. The native loop then sums
only up to the per-bin order — both more physical (modes with `n ≫ kr` carry no
energy) and faster. `order_per_freq` implements the `N ≈ k·r` rule:

```python
from smirgen import smir_generator, order_per_freq
c, r = 343, 0.105
h, H, _ = smir_generator(
    ..., sphRadius=r,
    N_harm=lambda f: order_per_freq(f, c, r, margin=2, n_max=30))
```

### Level relative to the source / direct sound

```python
from smirgen import relative_db
# ref="source": 0 dB = source's free-field level at 1 m (the generator's
#               native 1/R convention; |H| can exceed 1 from near-field/room gain).
# ref="direct": 0 dB = the direct arrival at the array (room colouration only).
freq, db, H = relative_db(ref="source", c=343, procFs=8000, sphLocation=[...],
                          s=[...], L=[...], beta=[0.7]*6, sphType="rigid",
                          sphRadius=0.042, mic=[[np.pi/4, np.pi/2]],
                          N_harm=20, nsample=2048, K=1, order=-1)
```

### Geometry plot

```python
from smirgen import plot_geometry
fig, ax = plot_geometry(L=[5,6,4], sphLocation=[1.6,4.05,1.7], s=[3.37,4.05,1.7],
                        sphRadius=0.042, mic=[[np.pi/4,np.pi/2],[np.pi/2,np.pi/2]],
                        src_ang=[0.2, 1.4])     # 2 panels: room + array close-up
```
Or `smir_generator(..., plot_geometry=True)`. It is a one-off inspection aid
(building the figure costs far more than the simulation) — keep it off in loops.
In Jupyter it displays inline without blocking.

## Fast dataset generation

For building datasets you usually call the generator many times. Two helpers
make this fast; both return results **identical** to `smir_generator`.

### CPU — `smir_generator_batch`

Each RIR is independent and the native loop releases the GIL, so a thread pool
scales almost linearly. Pass the fixed config as keyword arguments and a list of
per-sample overrides; results come back in input order.

```python
from smirgen import smir_generator_batch, order_per_freq
import numpy as np

freq   = np.fft.rfftfreq(2048, 1/16000)
N_harm = order_per_freq(freq, c=343, sph_radius=0.1, margin=10, n_max=30)

varying = [dict(s=src) for src in source_positions]      # only the source varies
rirs = smir_generator_batch(                              # list of h, input order
    varying, n_workers=8,
    c=343, procFs=16000, sphLocation=center, L=room, beta=0.3,
    sphType="rigid", sphRadius=0.1, mic=mic, N_harm=N_harm,
    nsample=2048, K=1, order=-1, HP=1, fmin=10)
# return_H=True -> list of (h, H, beta_hat) instead of just h
```

### GPU — `SmirArray` (PyTorch)

Specialised for the common case where the **room, array centre, microphones and
radius are fixed and only the source position varies**. Source-independent state
is precomputed once; the per-source spherical-harmonic sum runs as batched
matmuls plus a batched inverse FFT. Fast-path scope: spherical array
(open/rigid), **omnidirectional source**, **real reflection coefficients**.

```python
from smirgen import SmirArray, order_per_freq
import numpy as np

freq   = np.fft.rfftfreq(2048, 1/16000)
N_harm = order_per_freq(freq, c=343, sph_radius=0.1, margin=10, n_max=30)

arr = SmirArray(                       # build once for the fixed array/room
    c=343, procFs=16000, sphLocation=center, L=room, beta=0.3,
    sphType="rigid", sphRadius=0.1, mic=mic, N_harm=N_harm,
    nsample=2048, K=1, order=-1, HP=0, fmin=10,
    device="cuda", dtype="complex64")  # device="cuda:1" to pick a GPU

h = arr.generate(source_positions,     # (N, 3) array of source positions
                 source_batch=64,      # sources processed together (↑ throughput)
                 image_tile=8192)      # images per tile (caps peak memory)
# h:           (N, M, nsample)
# return_H=True -> (h, H) with H of shape (N, M, K*nsample/2+1)
# return_numpy=False -> leave results on-device as torch tensors
```

- `source_batch` and `image_tile` trade memory for speed; peak memory ≈
  `source_batch · k_total · image_tile` complex numbers. On `CUDA out of memory`,
  lower `image_tile` first. `arr.P` is the (pruned) candidate-image count.
- For multiple GPUs, build one `SmirArray` per `device="cuda:i"` and split the
  sources across them.

### Precision: `complex64` vs `complex128`

`SmirArray` matches `smir_generator` to **~1e-7** in `complex128` and **~1.5e-5**
in the default `complex64`. The float32 path is only safe up to spherical-harmonic
**order ≈ 25**; above that the rigid Hankel recurrence loses precision and
`complex64` silently degrades (tens of %). Two things push the required order up:

- **Large additive `N_harm` margin** (e.g. `margin=30` → order ~45): unnecessary
  for convergence and breaks `complex64`. `margin≈10` (order ~25) is converged
  for a mid-radius array and float32-safe.
- **Sources very close to the sphere surface** (near field): convergence needs a
  large *constant* order (~`ln(1/ε)/ln(R/r)` extra, with `R` the source distance
  and `r` the radius). A source ~1 cm off a 10 cm sphere needs order ~70 → use
  `complex128` (and `margin≈60`). Keep sources ≳2–3 cm off the surface to stay in
  the `complex64`-safe, low-order regime.

In short: **omni source, real coefficients, fixed array, sources not hugging the
surface → `complex64`; otherwise `complex128`.** `HP` only affects `h`, never `H`.

## Long RIRs — hybrid generation

The pure image-source method places every reflection exactly, but the image
count grows with the **cube of the RIR length** (`~ T60³·c³/V`). In a large,
reverberant room a long, fully geometric RIR is therefore expensive — the
bottleneck even on the GPU backend.

`smir_generator_hybrid` keeps the cheap, exact **early** part from the image
method (up to a bounded reflection order) and replaces the expensive **late
tail** with a fast statistical model: a multichannel diffuse-field noise whose
**inter-microphone coherence matches the array's diffuse coherence** (open
sphere: `sinc(k·d)`; rigid sphere: the mode-strength-weighted Legendre sum) and
whose energy decays at the room's T60. The two are crossfaded at the mixing
time.

```python
from smirgen import hybrid_params, smir_generator_hybrid
import numpy as np

mic = np.array([[0, np.pi/2], [2*np.pi/3, np.pi/2], [4*np.pi/3, np.pi/2], [0, 0.0]])

# Room dimensions + target T60 -> all the right parameters (pyroomacoustics-style).
kw = hybrid_params(L=[10, 8, 4], rt60=0.8, sphRadius=0.042,
                   procFs=16000, sphType="rigid")
# kw["info"] -> {'beta_hat', 'mix_time_ms', 'early_order', 'full_order', 'nsample'}
#   e.g. early_order=4  vs  full_order≈69 a full geometric RIR would need.

h, beta_hat = smir_generator_hybrid(
    sphLocation=[5, 4, 1.7], s=[2, 6, 1.5], mic=mic, seed=0, **kw)
# h: (M, nsample). ~260× faster than smir_generator(order=-1) for this room.
```

You can also call it directly; `early_order=None` (the default) is auto-derived
from the mixing time and room size, or set it yourself (larger → more of the RIR
is geometric / slower / more accurate). Other knobs: `mix_time` (crossover, s),
`tail_gain` (seam-level calibration), `xfade_ms`, `seed` (reproducible tails),
and `return_parts=True` to inspect the early RIR, tail and mixing sample.

> **Not sample-identical to a full image-source RIR.** The tail is a
> statistically-matched realisation — equal to the geometric tail only in its
> **energy-decay (T60)** and **spatial-coherence** statistics. That is the right
> trade-off for dereverberation / source-separation **training data**, where
> those statistics, not the exact late echo pattern, are what matter. The diffuse
> coherence-constrained noise follows Habets, Cohen & Gannot, *"Generating
> nonstationary multisensor signals under a spatial coherence constraint"*, JASA
> 124(5), 2008. Scalar T60 or 6 real reflection coefficients only (no
> angle-dependent walls in the tail).

## Conventions

- Angles are **(azimuth, inclination)** in radians, inclination from the **+z
  axis** (physics convention), matching the original `mysph2cart.m`. Helpers
  `smirgen.sph2cart` / `smirgen.cart2sph` are provided.
- `H` uses the free-field Green's function `≈ 1/R` for the direct path (the
  `1/(4π)` is folded out), so it is the pressure **relative to the source's
  level at 1 m** — values above 1 are physical (near field / room resonances).

## Repository layout

| Path | Purpose |
|------|---------|
| `smirgen/` | the package (generator, hybrid, torch_gen, coords, plotting, analysis) |
| `smir_loop.cpp` | pybind11 port of the MEX image-source core |
| `examples/example.py` | port of `run_smir_generator.m` |
| `tests/` | pytest correctness tests |
| `original_matlab/` | the upstream MATLAB/MEX sources (provenance, comparison) |
| `CODE_REVIEW.md` | review of the original code + fixes applied in the port |
| `CHANGELOG.md` | version history |

## Validation

The inner C++ loop is numerically identical to the original MEX implementation.
`tests/` checks the direct-path arrival time/amplitude, exact source/receiver
reciprocity, monotonic energy growth with reflection order, scalar-vs-array
`N_harm` equivalence, and that an anechoic direct-only run is 0 dB under
`relative_db`.

## License & attribution

MIT. This is a derivative work of the SMIR Generator
(© 2003–2015 E.A.P. Habets, International Audio Laboratories Erlangen) and
Habets' RIR Generator. The original MATLAB/MEX sources are included under
`original_matlab/`. See `LICENSE`.
