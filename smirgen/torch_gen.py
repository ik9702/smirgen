"""GPU (PyTorch) backend for fast dataset generation.

This is a fast-path reimplementation of :func:`smirgen.smir_generator` for the
common dataset case:

* spherical array (``sphType`` ``"open"`` or ``"rigid"``, ``sphRadius > 0``),
* **omnidirectional** source (``src_type='o'``),
* **real** reflection coefficients (``refl_coeff_ang_dep == 0``),

and is specialised for the workload where the room, array centre, microphones
and array radius are **fixed** and only the **source position** varies across a
dataset. Everything that does not depend on the source (mode strength, the
wavenumber grid, the image-source lattice and its reflection factors) is
precomputed once on the host; the per-source work — the spherical-Hankel /
Legendre expansion summed over image sources — runs on the GPU as batched
matrix multiplies, then a single batched inverse FFT.

The numerics mirror the native ``_smir_loop`` core exactly (same recurrences,
same per-frequency order truncation, same ``conj``/``irfft`` convention), so the
output matches :func:`smir_generator` to floating-point tolerance.

Typical use::

    from smirgen.torch_gen import SmirArray
    arr = SmirArray(c=343, procFs=16000, sphLocation=center, L=room, beta=0.3,
                    sphType="rigid", sphRadius=0.105, mic=mic, N_harm=N_harm,
                    nsample=2048, K=1, order=-1, HP=1, fmin=10, device="cuda")
    h = arr.generate(source_positions)        # (N, M, nsample) torch tensor

Requires PyTorch (CPU or CUDA). Import is lazy so the rest of ``smirgen`` works
without torch installed.
"""

import numpy as np

from .coords import sph2cart, cart2sph
from .generator import order_per_freq, _mode_strength_cached


def _build_shd_k_l(sphType, sphRadius, waveNr, n_per):
    """Precompute shd_k_l[K, Lmax+1] = 1j * modeStrength * waveNr, with the
    per-frequency order truncation baked in (entries with ``l > n_per[k]`` are
    zeroed so they contribute nothing to the harmonic sum)."""
    k_total = waveNr.shape[0]
    N_max = int(n_per.max())
    ms, N_max = _mode_strength_cached(sphType, sphRadius, waveNr, N_max)
    n_per = np.clip(n_per, 0, N_max).astype(int)
    shd = (1j * ms * waveNr[:, None]).astype(np.complex128)   # (K, N_max+1)
    # zero orders above the per-bin truncation
    ll = np.arange(N_max + 1)[None, :]
    shd[ll > n_per[:, None]] = 0
    return shd, N_max


class SmirArray:
    """Precomputed, source-independent state for a fixed array/room.

    Construct once, then call :meth:`generate` with any set of source positions.
    """

    def __init__(self, c, procFs, sphLocation, L, beta, sphType, sphRadius, mic,
                 N_harm, nsample, K=1, order=-1, HP=0, fmin=0.0,
                 device="cuda", dtype="complex64"):
        import torch

        if sphType not in ("open", "rigid"):
            raise ValueError("sphType must be 'open' or 'rigid'.")
        if sphRadius <= 0:
            raise ValueError("torch_gen requires a spherical array (sphRadius>0).")

        self.torch = torch
        self.device = torch.device(device)
        cdt = {"complex64": torch.complex64, "complex128": torch.complex128}[dtype]
        rdt = torch.float32 if cdt == torch.complex64 else torch.float64
        self.cdt, self.rdt = cdt, rdt
        self.c = float(c)
        self.procFs = float(procFs)
        self.nsample = int(nsample)
        self.K = int(K)
        self.HP = int(HP)
        self.N_FFT = self.K * self.nsample
        self.sphRadius = float(sphRadius)

        sphLocation = np.asarray(sphLocation, float).reshape(3)
        L = np.asarray(L, float).reshape(3)
        beta = np.asarray(beta, float).reshape(-1)

        # ---- reflection coefficients (scalar T60 -> 6 walls) ----------------
        if beta.size == 1:
            V = L[0] * L[1] * L[2]
            S = 2 * (L[0] * L[2] + L[1] * L[2] + L[0] * L[1])
            alfa = 24 * V * np.log(10) / (c * S * beta[0])
            if alfa > 1:
                raise ValueError("Cannot derive reflection coefficients from "
                                 "the given T60/room dimensions.")
            beta = np.full(6, np.sqrt(1 - alfa))
        elif beta.size != 6:
            raise ValueError("beta must be a scalar (T60) or length-6 vector.")
        self.beta6 = beta

        # ---- frequency grid / wavenumbers (host) ----------------------------
        k_total = self.N_FFT // 2 + 1
        kk = np.arange(k_total)
        freq = kk * self.procFs / self.N_FFT
        freq_eff = np.maximum(freq, fmin) if fmin and fmin > 0 else freq
        waveNr = 2 * np.pi * freq_eff / c

        if callable(N_harm):
            n_per = np.asarray(N_harm(freq), int).reshape(-1)
        elif np.ndim(N_harm) > 0:
            n_per = np.asarray(N_harm, int).reshape(-1)
        else:
            n_per = np.full(k_total, int(N_harm))
        if n_per.size != k_total:
            raise ValueError(f"Per-frequency N_harm must have length {k_total}.")

        shd_k_l, self.N_max = _build_shd_k_l(sphType, sphRadius, waveNr, n_per)

        # ---- image-source lattice (source-independent part) -----------------
        cTs = c / self.procFs
        self.cTs = cTs
        s_dummy = sphLocation  # only need lattice extent from nsample & L_samp
        L_samp = L / cTs
        r_samp = sphLocation / cTs
        n1 = int(np.ceil(self.nsample / (2 * L_samp[0])))
        n2 = int(np.ceil(self.nsample / (2 * L_samp[1])))
        n3 = int(np.ceil(self.nsample / (2 * L_samp[2])))

        mx, my, mz, q, j, kp = np.meshgrid(
            np.arange(-n1, n1 + 1), np.arange(-n2, n2 + 1),
            np.arange(-n3, n3 + 1), [0, 1], [0, 1], [0, 1], indexing="ij")
        mx, my, mz, q, j, kp = (a.ravel() for a in (mx, my, mz, q, j, kp))

        # hu3 = (1-2q) s0 - r0 + 2 mx L0  ->  sign * s + const   (sample units)
        self._sign = np.stack([1 - 2 * q, 1 - 2 * j, 1 - 2 * kp], 1).astype(float)
        self._const = np.stack([
            -r_samp[0] + 2 * mx * L_samp[0],
            -r_samp[1] + 2 * my * L_samp[1],
            -r_samp[2] + 2 * mz * L_samp[2]], 1).astype(float)

        # reflection factor (omni, real coeffs): product of wall coeffs.
        reflbeta = (beta[0] ** np.abs(mx - q) * beta[1] ** np.abs(mx) *
                    beta[2] ** np.abs(my - j) * beta[3] ** np.abs(my) *
                    beta[4] ** np.abs(mz - kp) * beta[5] ** np.abs(mz))
        # reflection-order constraint
        if order != -1:
            okorder = (np.abs(2 * mx - q) + np.abs(2 * my - j) +
                       np.abs(2 * mz - kp)) <= order
            reflbeta = reflbeta * okorder
        reflbeta = reflbeta.astype(float)

        # Prune candidate images that can never satisfy fdist < nsample for ANY
        # source inside the room. Per axis the image term is sign*s + const with
        # s in [0, L_samp]; its minimum |.| (0 if the interval spans 0) gives the
        # smallest possible image distance over all in-room sources. This is an
        # exact prune (no valid image is dropped) and typically removes the far
        # lattice corners that dominate the candidate count.
        lo = np.minimum(self._const, self._const + self._sign * L_samp)
        hi = np.maximum(self._const, self._const + self._sign * L_samp)
        min_term = np.where((lo <= 0) & (hi >= 0), 0.0,
                            np.minimum(np.abs(lo), np.abs(hi)))
        min_dist = np.sqrt((min_term ** 2).sum(1))
        keep = np.floor(min_dist + sphRadius / cTs) < self.nsample
        keep &= reflbeta != 0          # also drop order-excluded images
        self._sign = self._sign[keep]
        self._const = self._const[keep]
        self._reflbeta = reflbeta[keep]

        # microphone unit-cartesian positions
        mx_, my_, mz_ = sph2cart(np.asarray(mic, float)[:, 0],
                                 np.asarray(mic, float)[:, 1], 1.0)
        mic_pos = np.column_stack([mx_, my_, mz_]).astype(float)
        self.M = mic_pos.shape[0]

        # ---- frequency ordering by required spherical-harmonic order --------
        # At frequency k only orders l <= n_per[k] contribute. Sorting the bins
        # by n_per (descending) makes "orders >= l are active" a contiguous top
        # slice whose length ``counts[l]`` shrinks as l grows, so the per-order
        # Hankel recurrence and contraction touch only the still-active bins
        # (low frequencies drop out early). This both avoids the wasted work on
        # masked entries and removes the need for an inf*0->NaN guard.
        n_per = np.clip(n_per, 0, self.N_max).astype(int)
        perm = np.argsort(-n_per, kind="stable")          # descending by order
        n_per_sorted = n_per[perm]
        counts = [int((n_per_sorted >= l).sum()) for l in range(self.N_max + 1)]
        self.counts = counts
        shd_k_l = shd_k_l[perm]                            # rows -> sorted order
        waveNr = waveNr[perm]
        self._perm = perm

        # ---- move constants to the device ----------------------------------
        t = torch
        self.shd_k_l = t.as_tensor(shd_k_l, dtype=cdt, device=self.device)      # (K, L+1)
        self.waveNr = t.as_tensor(waveNr, dtype=rdt, device=self.device)        # (K,) sorted
        self.perm = t.as_tensor(perm, dtype=t.long, device=self.device)         # (K,)
        self.mic_pos = t.as_tensor(mic_pos, dtype=rdt, device=self.device)      # (M, 3)
        self.sign = t.as_tensor(self._sign, dtype=rdt, device=self.device)      # (P, 3)
        self.const = t.as_tensor(self._const, dtype=rdt, device=self.device)    # (P, 3)
        self.reflbeta = t.as_tensor(self._reflbeta, dtype=rdt, device=self.device)  # (P,)
        self.l_factor = t.as_tensor(2 * np.arange(self.N_max + 1) + 1.0,
                                    dtype=rdt, device=self.device)              # (L+1,)
        self.k_total = k_total
        self.P = self.sign.shape[0]

        # HP filter coefficients (applied on host at the end)
        self._hp_ba = None
        if self.HP == 1:
            from scipy.signal import butter
            self._hp_ba = butter(4, 50 / (self.procFs / 2), btype="high")

    # -----------------------------------------------------------------------
    def generate(self, sources, source_batch=16, image_tile=4096,
                 return_numpy=True, return_H=False):
        """Generate RIRs for many source positions.

        Parameters
        ----------
        sources : (N, 3) array_like
            Source positions (m).
        source_batch : int
            Number of sources processed together on the GPU per step.
        image_tile : int
            Number of candidate image sources processed per tile (bounds the
            peak memory: ~ ``source_batch * k_total * image_tile`` complex).
        return_numpy : bool
            Return a NumPy array (default) or leave the result on-device as a
            torch tensor.
        return_H : bool
            If True, also return the one-sided complex transfer function ``H``
            of shape ``(N, M, K*nsample/2 + 1)`` (matching :func:`smir_generator`
            output ``H``). Default False.

        Returns
        -------
        h : (N, M, nsample) array
            Room impulse responses, matching :func:`smir_generator` output ``h``.
        H : (N, M, K*nsample/2 + 1) array, optional
            Returned only when ``return_H=True``.
        """
        t = self.torch
        sources = np.asarray(sources, float).reshape(-1, 3)
        N = sources.shape[0]
        s_samp = t.as_tensor(sources / self.cTs, dtype=self.rdt,
                             device=self.device)                      # (N, 3)
        sph_samp = self.sphRadius / self.cTs
        out = t.empty((N, self.M, self.nsample), dtype=self.rdt,
                      device=self.device)
        H_out = (t.empty((N, self.M, self.k_total), dtype=self.cdt,
                         device=self.device) if return_H else None)

        for b0 in range(0, N, source_batch):
            sb = s_samp[b0:b0 + source_batch]                          # (S, 3)
            S = sb.shape[0]
            H = self._accumulate(sb, S, sph_samp, image_tile)          # (S, M, K) complex
            H = t.nan_to_num(H)
            H = t.conj(H)
            if return_H:
                H_out[b0:b0 + S] = H
            h = t.fft.irfft(H, n=self.N_FFT, dim=-1)[..., :self.nsample]
            out[b0:b0 + S] = h

        if self.HP == 1:
            out = self._apply_hp(out)

        if return_numpy:
            out = out.detach().cpu().numpy()
            if return_H:
                H_out = H_out.detach().cpu().numpy()
        return (out, H_out) if return_H else out

    # -----------------------------------------------------------------------
    def _accumulate(self, sb, S, sph_samp, image_tile):
        """H[S, M, K] = sum over image sources of the harmonic expansion.

        Frequencies are held in descending-order-of-``n_per`` (sorted at
        construction). At harmonic order ``l`` only the first ``counts[l]``
        sorted bins are still active, so the recurrence and contraction operate
        on a shrinking prefix. H is un-sorted back to the natural bin order
        before returning.
        """
        t = self.torch
        # image positions (sample units): hu[S,P,3] = sign[P,3]*s[S,3] + const[P,3]
        hu = self.sign[None] * sb[:, None, :] + self.const[None]       # (S, P, 3)
        dist = t.linalg.norm(hu, dim=2)                                # (S, P)
        valid = (t.floor(dist + sph_samp) < self.nsample) & (dist > 1e-9)
        u = hu / dist.clamp_min(1e-30)[..., None]                      # (S, P, 3)
        R_norm = dist * self.cTs                                       # (S, P) physical
        cosang = (u @ self.mic_pos.T).clamp(-1.0, 1.0)                 # (S, P, M)
        reflb = self.reflbeta[None] * valid.to(self.rdt)               # (S, P)

        H = t.zeros((S, self.M, self.k_total), dtype=self.cdt,
                    device=self.device)                               # sorted bins
        waveNr = self.waveNr                                           # (K,) sorted
        I = t.tensor(1j, dtype=self.cdt, device=self.device)
        counts = self.counts

        for p0 in range(0, self.P, image_tile):
            p1 = min(p0 + image_tile, self.P)
            Rt = R_norm[:, p0:p1]                                      # (S, T)
            cb = cosang[:, p0:p1, :]                                   # (S, T, M)
            rb = reflb[:, p0:p1]                                       # (S, T)

            # z[S, K, T] = waveNr[K] (x) R_norm[S, T]  (sorted bin order). z is
            # real, so e^{iz}=cos z + i sin z and 1/z is a real reciprocal --
            # both cheaper than the complex exp / complex division.
            zr = waveNr[None, :, None] * Rt[:, None, :]                 # (S, K, T) real
            eiz = t.complex(t.cos(zr), t.sin(zr))
            invz = (1.0 / zr).to(self.cdt)
            # spherical Hankel h_0, h_1 (first kind)
            h_prev = eiz * invz / I                                    # h_0 = e^{iz}/(iz)
            h_curr = -I * (-I * invz + invz * invz) * eiz              # h_1

            # Legendre P_0, P_1 over (S, T, M)
            p_prev = t.ones_like(cb)                                   # P_0 = 1
            p_curr = cb.clone()                                        # P_1 = x

            self._contract(H, 0, h_prev, p_prev, rb, counts[0])
            if self.N_max >= 1:
                self._contract(H, 1, h_curr, p_curr, rb, counts[1])

            for l in range(2, self.N_max + 1):
                na = counts[l]                                         # active bins
                if na == 0:
                    break
                a = 2 * l - 1
                # recurrence on the active prefix only
                h_next = a * invz[:, :na] * h_curr[:, :na] - h_prev[:, :na]
                p_next = ((2 * l - 1) * cb * p_curr - (l - 1) * p_prev) / l
                self._contract(H, l, h_next, p_next, rb, na)
                h_prev, h_curr = h_curr[:, :na], h_next
                p_prev, p_curr = p_curr, p_next

        # un-sort: H[..., perm] = H_sorted
        H_out = t.empty_like(H)
        H_out.index_copy_(2, self.perm, H)
        return H_out

    def _contract(self, H, l, hank_l, p_l, rb, na):
        """Add order ``l``'s contribution into ``H[:, :, :na]`` in place.

        hank_l: (S, na, T) complex   p_l: (S, T, M) real   rb: (S, T) real
        """
        if na == 0:
            return
        t = self.torch
        # A[S, M, T] = (2l+1) * P_l * reflbeta
        A = (self.l_factor[l] * p_l * rb[:, :, None]).transpose(1, 2).to(self.cdt)
        # Contract the image-tile axis t first, then apply the per-frequency
        # mode-strength factor on the small (S, M, na) result instead of on the
        # large (S, na, T) Hankel tensor (all bins here are active -> no mask).
        contrib = t.einsum("smt,snt->smn", A, hank_l)                   # (S, M, na)
        H[:, :, :na] += contrib * self.shd_k_l[:na, l][None, None, :]

    def _apply_hp(self, h):
        """4th-order 50 Hz Butterworth high-pass, matching scipy.lfilter.

        The 50 Hz / Nyquist cutoff puts the IIR poles very close to z=1, so the
        recurrence is run in float64 regardless of the compute dtype (float32
        would accumulate catastrophic error here); cost is negligible.
        """
        b, a = self._hp_ba
        t = self.torch
        N, M, L = h.shape
        x = h.reshape(N * M, L).to(t.float64)
        b = t.as_tensor(b, dtype=t.float64, device=h.device)
        a = t.as_tensor(a, dtype=t.float64, device=h.device)
        # Direct-form II transposed, sample-by-sample (filter order is small).
        na, nb = a.shape[0], b.shape[0]
        zlen = max(na, nb) - 1
        zi = t.zeros((N * M, zlen), dtype=t.float64, device=h.device)
        y = t.empty_like(x)
        for n in range(L):
            xn = x[:, n]
            yn = b[0] * xn + zi[:, 0]
            for k in range(1, zlen + 1):
                bk = b[k] if k < nb else 0.0
                ak = a[k] if k < na else 0.0
                if k < zlen:
                    zi[:, k - 1] = bk * xn + zi[:, k] - ak * yn
                else:
                    zi[:, k - 1] = bk * xn - ak * yn
            y[:, n] = yn
        return y.reshape(N, M, L).to(self.rdt)
