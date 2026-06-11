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
| `smirgen/` | the package (generator, coords, plotting, analysis) |
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
