"""Basic correctness tests for smirgen.

Run with: pytest
"""

import numpy as np
import pytest

from smirgen import (smir_generator, smir_generator_batch, order_per_freq,
                     relative_db)

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


def test_batch_matches_sequential():
    """smir_generator_batch returns, in order, exactly the sequential results."""
    common = dict(c=C, procFs=FS, sphLocation=REC, L=L, beta=0.3,
                  sphType="rigid", sphRadius=0.042,
                  mic=[[np.pi / 4, np.pi / 2], [np.pi / 2, np.pi / 2]],
                  N_harm=15, nsample=1024, K=1, order=-1, HP=1)
    varying = [dict(s=[3.37 + 0.05 * i, 4.05, 1.7]) for i in range(6)]
    seq = [smir_generator(**{**common, **v})[0] for v in varying]
    bat = smir_generator_batch(varying, n_workers=4, **common)
    assert len(bat) == len(varying)
    for a, b in zip(seq, bat):
        assert np.array_equal(a, b)


def test_batch_return_H_and_edge_cases():
    common = dict(c=C, procFs=FS, sphLocation=REC, L=L, beta=ANECHOIC,
                  sphType="open", sphRadius=0.0, mic=[[0, 0]], N_harm=0,
                  nsample=512, K=1, order=0, HP=0)
    assert smir_generator_batch([], **common) == []
    one = smir_generator_batch([dict(s=SRC)], **common)
    assert len(one) == 1 and one[0].ndim == 2
    tup = smir_generator_batch([dict(s=SRC)], return_H=True, **common)
    assert len(tup[0]) == 3


@pytest.mark.parametrize("sphType", ["open", "rigid"])
@pytest.mark.parametrize("dtype,tol", [("complex128", 1e-5), ("complex64", 5e-4)])
def test_torch_backend_matches_reference(sphType, dtype, tol):
    """SmirArray (GPU backend, run here on CPU) matches smir_generator."""
    torch = pytest.importorskip("torch")
    from smirgen import SmirArray
    r = 0.042
    mic = [[np.pi / 4, np.pi / 2], [np.pi / 2, np.pi / 3]]
    freq = np.fft.rfftfreq(256, 1 / FS)
    N_harm = order_per_freq(freq, C, r, margin=6, n_max=20)
    srcs = [[3.37, 4.05, 1.7], [2.5, 3.0, 2.1], [4.0, 5.0, 1.2]]
    common = dict(c=C, procFs=FS, sphLocation=REC, L=L, beta=0.3,
                  sphType=sphType, sphRadius=r, mic=mic, N_harm=N_harm,
                  nsample=256, K=1, order=-1, HP=1, fmin=5)
    ref = np.stack([smir_generator(s=s, **common)[0] for s in srcs])
    refH = np.stack([smir_generator(s=s, **common)[1] for s in srcs])
    arr = SmirArray(device="cpu", dtype=dtype, **common)
    got, gotH = arr.generate(srcs, source_batch=2, image_tile=600, return_H=True)
    assert got.shape == ref.shape
    assert gotH.shape == refH.shape
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < tol
    assert np.linalg.norm(gotH - refH) / np.linalg.norm(refH) < tol
