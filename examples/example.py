"""Port of run_smir_generator.m - three example impulse responses.

Run from this directory after building the extension:
    python example.py
"""

import numpy as np

from smirgen import smir_generator, cart2sph

# ---- Setup ----------------------------------------------------------------
procFs = 8000          # sampling frequency (Hz)
c = 343                # speed of sound (m/s)
nsample = 2 * 1024     # RIR length
N_harm = 30            # max spherical-harmonic order
K = 1                  # oversampling factor

L = [5, 6, 4]                  # room dimensions (x, y, z) in m
sphLocation = [1.6, 4.05, 1.7]  # array centre
s = [3.37, 4.05, 1.7]           # source location

HP = 1
src_type = "o"
az, inc, _ = cart2sph(sphLocation[0] - s[0],
                      sphLocation[1] - s[1],
                      sphLocation[2] - s[2])
src_ang = [az, inc]            # towards the receiver

mic = [[np.pi / 4, np.pi], [np.pi / 2, np.pi]]

# ---- Example 1: rigid sphere, real reflection coefficients ----------------
h1, H1, _ = smir_generator(c, procFs, sphLocation, s, L, beta=0.3,
                           sphType="rigid", sphRadius=0.042, mic=mic,
                           N_harm=N_harm, nsample=nsample, K=K, order=6,
                           refl_coeff_ang_dep=0, HP=HP, src_type=src_type,
                           src_ang=src_ang)
print(f"Example 1 (rigid, real beta):  h1 {h1.shape}, H1 {H1.shape}")

# ---- Example 2: rigid sphere, angle-dependent reflection coefficients ------
sigma = 1.5e4 * np.ones(6)     # effective flow resistivity
h2, H2, _ = smir_generator(c, procFs, sphLocation, s, L, beta=sigma,
                           sphType="rigid", sphRadius=0.042, mic=mic,
                           N_harm=N_harm, nsample=nsample, K=K, order=6,
                           refl_coeff_ang_dep=1, HP=HP, src_type=src_type,
                           src_ang=src_ang)
print(f"Example 2 (rigid, Komatsu):    h2 {h2.shape}, H2 {H2.shape}")

# ---- Example 3: single microphone at the centre ---------------------------
h3, H3, beta_hat = smir_generator(c, procFs, sphLocation, s, L, beta=0.3,
                                  sphType="open", sphRadius=0.0, mic=[[0, 0]],
                                  N_harm=N_harm, nsample=nsample, K=K, order=6,
                                  refl_coeff_ang_dep=0, HP=HP, src_type=src_type,
                                  src_ang=src_ang)
print(f"Example 3 (single mic):        h3 {h3.shape}, H3 {H3.shape}, "
      f"beta_hat={beta_hat:.4f}")

# ---- 3D geometry plot -----------------------------------------------------
try:
    from smirgen import plot_geometry
    fig_geo, _ = plot_geometry(L, sphLocation, s, sphRadius=0.042,
                               mic=mic, src_ang=src_ang, sphType="rigid")
    fig_geo.savefig("example_geometry.png", dpi=120, bbox_inches="tight")
    print("Saved geometry to example_geometry.png")
    # Equivalent one-liner during a run:
    #   smir_generator(..., plot_geometry=True)
except ImportError:
    print("(matplotlib not installed - skipping geometry plot)")

# ---- Optional RIR plotting (only if matplotlib is available) --------------
try:
    import matplotlib.pyplot as plt

    t = np.arange(nsample) / procFs
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(t, h1[0], "r")
    ax[0].set_title("RIR at mic 1 (real refl. coeff.)")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(t, h2[0], "r")
    ax[1].set_title("RIR at mic 1 (angle-dependent refl. coeff.)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig("example_rir.png", dpi=120)
    print("Saved plot to example_rir.png")
except ImportError:
    print("(matplotlib not installed - skipping plot)")
