"""Basic correctness tests for smirgen.

Run with: pytest
"""

import numpy as np
import pytest

from smirgen import smir_generator, order_per_freq, relative_db

C = 343.0
FS = 8000.0
REC = [1.6, 4.05, 1.7]
SRC = [3.37, 4.05, 1.7]
L = [5, 6, 4]
ANECHOIC = [0, 0, 0, 0, 0, 0]


def _single_mic(beta, order, nsample=2048, **kw):
    return smir_generator(c=C, procFs=FS, sphLocation=REC, s=SRC, L=L, beta=beta,
                          sphType="open", sphRadius=0.0, mic=[[0, 0]], N_harm=0,
                          nsample=nsample, K=1, order=order, HP=0, **kw)


def test_direct_path_timing_and_amplitude():
    """Free-field direct path: peak at R/c and amplitude ~1/R."""
    R = np.linalg.norm(np.array(REC) - np.array(SRC))
    h, H, _ = _single_mic(ANECHOIC, order=0)
    h = h[0]
    assert abs(np.argmax(np.abs(h)) - R / C * FS) <= 1
    assert np.abs(H[0, 5]) == pytest.approx(1.0 / R, rel=1e-3)


def test_reciprocity():
    """Swapping source and receiver gives an identical free-field RIR."""
    h1, _, _ = _single_mic(ANECHOIC, order=0)
    h2, _, _ = smir_generator(c=C, procFs=FS, sphLocation=SRC, s=REC, L=L,
                              beta=ANECHOIC, sphType="open", sphRadius=0.0,
                              mic=[[0, 0]], N_harm=0, nsample=2048, K=1,
                              order=0, HP=0)
    assert np.allclose(h1[0], h2[0])


def test_energy_increases_with_order():
    energies = []
    for o in (0, 1, 2, -1):
        h, _, _ = _single_mic([0.7] * 6, order=o, nsample=2048)
        energies.append(np.sum(h[0] ** 2))
    assert all(b > a for a, b in zip(energies, energies[1:]))


def test_scalar_equals_uniform_array():
    """Per-frequency N_harm equal to a constant matches the scalar case."""
    kw = dict(c=C, procFs=FS, sphLocation=REC, s=SRC, L=L, beta=[0.7] * 6,
              sphType="rigid", sphRadius=0.042, mic=[[np.pi / 4, np.pi / 2]],
              nsample=1024, K=1, order=4, HP=0)
    _, H_scalar, _ = smir_generator(N_harm=20, **kw)
    k_total = 1024 // 2 + 1
    _, H_array, _ = smir_generator(N_harm=np.full(k_total, 20), **kw)
    assert np.allclose(np.nan_to_num(H_scalar), np.nan_to_num(H_array))


def test_order_per_freq_monotonic():
    freq = np.linspace(0, FS / 2, 200)
    n = order_per_freq(freq, C, 0.105, margin=2, n_max=30)
    assert n[0] <= n[-1]
    assert n.max() <= 30 and n.min() >= 0


def test_fmin_fills_dc_bin():
    kw = dict(c=C, procFs=FS, sphLocation=REC, s=SRC, L=L, beta=[0.5] * 6,
              sphType="rigid", sphRadius=0.042, mic=[[0, 0]], N_harm=10,
              nsample=512, K=1, order=2, HP=0)
    _, H0, _ = smir_generator(fmin=0.0, **kw)
    _, H1, _ = smir_generator(fmin=10.0, **kw)
    assert np.abs(H0[0, 0]) == 0.0           # DC singular -> zeroed
    assert np.isfinite(H1[0, 0]) and np.abs(H1[0, 0]) > 0


def test_relative_db_anechoic_direct_is_zero():
    """Direct-referenced dB of an anechoic direct-only run is ~0 dB."""
    freq, db, _ = relative_db(ref="direct", c=C, procFs=FS, sphLocation=REC,
                              s=SRC, L=L, beta=ANECHOIC, sphType="rigid",
                              sphRadius=0.042, mic=[[np.pi / 4, np.pi / 2]],
                              N_harm=15, nsample=1024, K=1, order=0)
    assert np.nanmax(np.abs(db)) < 1e-6
