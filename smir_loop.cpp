/*
 * smir_loop.cpp
 *
 * pybind11 port of `smir_generator_loop.cpp` (the MEX core of the SMIR
 * generator).  The numerical inner loop is kept byte-for-byte identical to the
 * original MATLAB MEX implementation; only the I/O glue (mxGetPr / mxCreate...)
 * was replaced by pybind11 / NumPy buffers.
 *
 * Original algorithm/code:
 *   D. P. Jarrett, E. A. P. Habets, M. R. P. Thomas, P. A. Naylor,
 *   "Simulating room impulse responses for spherical microphone arrays",
 *   ICASSP 2011.  C++ loop + directivity + angle dependent reflection by
 *   S. Braun.  Copyright (C) 2015 International Audio Laboratories Erlangen.
 *
 * All matrices are passed in column-major (Fortran) order so that the index
 * arithmetic below matches the original MEX code exactly.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <complex>
#include <vector>
#include <numeric>   // inner_product
#include <cmath>
#include <string>

namespace py = pybind11;
using std::complex;
using std::vector;

static const complex<double> I = complex<double>(0, 1);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Helper functions (identical to the original MEX implementation)
// ---------------------------------------------------------------------------

static complex<double> refl_factor_komatsu(double phi_img, int N_FFT, double Fs,
                                           double beta, int idx)
{
    double freq;
    complex<double> Z, K, A, B, R;
    if (idx == 0)
        R = complex<double>(1.0, 0.0);
    else {
        freq = idx * (Fs / N_FFT);
        Z = 1.0 + 0.00027 * pow((2 - log(freq / beta)), 6.2)
              + I * 0.0047 * pow((2 - log(freq / beta)), 4.1);   // conj(Z)
        K = 0.0069 * pow((2 - log(freq / beta)), 4.1)
              - I * (1.0 + 0.0004 * pow((2 - log(freq / beta)), 6.2)); // conj(K)
        A = sin(phi_img) - pow(Z, -1) * pow(1.0 - pow(K, -2) * pow(cos(phi_img), 2), 0.5);
        B = sin(phi_img) + pow(Z, -1) * pow(1.0 - pow(K, -2) * pow(cos(phi_img), 2), 0.5);
        R = A / B;
    }
    return R;
}

static void normalize(double* arr, int length)
{
    double norm = 0;
    for (int ii = 0; ii < length; ii++) norm += pow(arr[ii], 2);
    norm = sqrt(norm);
    for (int ii = 0; ii < length; ii++) arr[ii] = arr[ii] / norm;
}

static void sphbesselh(int max_nu, double Z, vector<complex<double> >& output)
{
    output[0] = exp(I * Z) / (I * Z);
    if (max_nu < 1) return;
    output[1] = -I * (-I / Z + 1.0 / (Z * Z)) * exp(I * Z);
    for (int nu = 2; nu <= max_nu; nu++)
        output[nu] = (2.0 * nu - 1) / Z * output[nu - 1] - output[nu - 2];
}

static void legendre(int max_n, double X, vector<double>& output)
{
    output[0] = 1;
    if (max_n < 1) return;
    output[1] = X;
    for (int n = 2; n <= max_n; n++)
        output[n] = (float)(2 * n - 1) / n * X * output[n - 1]
                  - (float)(n - 1) / n * output[n - 2];
}

static double src_directivity(double* vect1, double* vect2, char src_type)
{
    if (src_type == 'b' || src_type == 'c' || src_type == 's' || src_type == 'h') {
        double strength, alpha = 1.0, vartheta, init = 0.0;
        switch (src_type) {
            case 'b': alpha = 0;    break;
            case 'h': alpha = 0.25; break;
            case 'c': alpha = 0.5;  break;
            case 's': alpha = 0.75; break;
        }
        normalize(vect1, 3);
        normalize(vect2, 3);
        vartheta = std::inner_product(vect1, vect1 + 3, vect2, init);  // cos(theta)
        strength = alpha + (1 - alpha) * vartheta;
        return strength;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Main entry point.  Returns the complex room transfer function H (M x k_total).
// ---------------------------------------------------------------------------

static py::array_t< complex<double> > smir_loop(
        double c, double fs,
        py::array_t<double, py::array::f_style | py::array::forcecast> rr,
        py::array_t<double, py::array::f_style | py::array::forcecast> ss,
        py::array_t<double, py::array::f_style | py::array::forcecast> LL,
        py::array_t<double, py::array::f_style | py::array::forcecast> beta_in,
        int nsamples, int order, int K,
        py::array_t<complex<double>, py::array::f_style | py::array::forcecast> shd_k_l,
        py::array_t<double, py::array::f_style | py::array::forcecast> shd_angle_l_all,
        py::array_t<double, py::array::f_style | py::array::forcecast> mic_pos_in,
        double sphRadius,
        py::array_t<double, py::array::f_style | py::array::forcecast> waveNr_in,
        int refl_coeff_ang_dep,
        py::array_t<double, py::array::f_style | py::array::forcecast> src_ang_in,
        std::string src_type_str,
        py::array_t<int, py::array::f_style | py::array::forcecast> n_harm_per_k)
{
    // ---- raw pointers -----------------------------------------------------
    const double* rr_p   = rr.data();
    const double* ss_p   = ss.data();
    const double* LL_p   = LL.data();
    const double* beta   = beta_in.data();
    const complex<double>* shd_k_l_p = shd_k_l.data();   // k_total x (N_harm+1) col-major
    const double* shd_angle_l_dependent_all_sources = shd_angle_l_all.data();
    const double* mic_pos = mic_pos_in.data();           // M x 3 col-major
    const double* waveNr  = waveNr_in.data();
    const double* src_ang = src_ang_in.data();
    const char    src_type0 = src_type_str.empty() ? 'o' : src_type_str[0];
    const int*    nhpk = n_harm_per_k.data();            // per-frequency max order

    const int k_total = (int) shd_k_l.shape(0);
    const int N_harm  = (int) shd_k_l.shape(1) - 1;      // global maximum order
    const int M       = (int) mic_pos_in.shape(0);
    const int N_FFT   = K * nsamples;

    // ---- output H (M x k_total, complex, column-major) --------------------
    std::vector<py::ssize_t> H_shape   = {(py::ssize_t) M, (py::ssize_t) k_total};
    std::vector<py::ssize_t> H_strides = {(py::ssize_t) sizeof(complex<double>),
                                          (py::ssize_t) sizeof(complex<double>) * M};
    py::array_t< complex<double> > H(H_shape, H_strides);
    complex<double>* Hd = H.mutable_data();
    for (py::ssize_t n = 0; n < (py::ssize_t) M * k_total; n++) Hd[n] = complex<double>(0, 0);

    // ---- scratch ----------------------------------------------------------
    double tmp_angle;
    double refl_angles[6];
    complex<double> R_p_plus_R_m_beta;
    complex<double> Q[6];
    vector<double>           legendre_out(N_harm + 1);
    vector< complex<double> > sphbesselh_out(N_harm + 1);
    vector< complex<double> > shd_k_l_dependent(k_total * (N_harm + 1));
    vector<double>            shd_angle_l_dependent((N_harm + 1) * M);
    // Per-image reflection*directivity factor, indexed by frequency bin. Used
    // only for angle-dependent reflection coefficients (otherwise it is a
    // single scalar). Hoisted out of the (mic x freq) accumulation loops.
    vector< complex<double> > reflbeta_k(k_total);

    const double cTs = c / fs;
    double r[3], s[3], L[3], hu[6];
    double dist;
    int    fdist;

    s[0] = ss_p[0] / cTs; s[1] = ss_p[1] / cTs; s[2] = ss_p[2] / cTs;
    L[0] = LL_p[0] / cTs; L[1] = LL_p[1] / cTs; L[2] = LL_p[2] / cTs;
    r[0] = rr_p[0] / cTs; r[1] = rr_p[1] / cTs; r[2] = rr_p[2] / cTs;

    int n1 = (int) ceil(nsamples / (2 * L[0]));
    int n2 = (int) ceil(nsamples / (2 * L[1]));
    int n3 = (int) ceil(nsamples / (2 * L[2]));

    // Release the GIL during the heavy numerical loop.
    {
        py::gil_scoped_release release;

        for (int mx = -n1; mx <= n1; mx++) {
            hu[0] = 2 * mx * L[0];
            for (int my = -n2; my <= n2; my++) {
                hu[1] = 2 * my * L[1];
                for (int mz = -n3; mz <= n3; mz++) {
                    hu[2] = 2 * mz * L[2];
                    for (int q = 0; q <= 1; q++) {
                        hu[3] = (1 - 2 * q) * s[0] - r[0] + hu[0];
                        for (int j = 0; j <= 1; j++) {
                            hu[4] = (1 - 2 * j) * s[1] - r[1] + hu[1];
                            for (int k = 0; k <= 1; k++) {
                                hu[5] = (1 - 2 * k) * s[2] - r[2] + hu[2];

                                dist = sqrt(pow(hu[3], 2) + pow(hu[4], 2) + pow(hu[5], 2));
                                double R_p_plus_R_m[3] = {hu[3] * cTs, hu[4] * cTs, hu[5] * cTs};
                                double R_p_plus_R_m_norm = dist * cTs;

                                if (abs(2 * mx - q) + abs(2 * my - j) + abs(2 * mz - k) <= order
                                        || order == -1) {
                                    fdist = (int) floor(dist + (sphRadius / cTs));
                                    if (fdist < nsamples) {
                                        normalize(R_p_plus_R_m, 3);
                                        for (int jj = 0; jj < 6; jj++) {
                                            double wall_normal[3] = {0, 0, 0};
                                            wall_normal[jj / 2] = 1;
                                            double init = 0.0;
                                            double dotprod = std::inner_product(
                                                R_p_plus_R_m, R_p_plus_R_m + 3, wall_normal, init);
                                            refl_angles[jj] = std::abs((M_PI / 2) - acos(dotprod));
                                        }

                                        double look_dir_mir[3] = {
                                            (pow(-1.0, q)) * src_ang[0],
                                            (pow(-1.0, j)) * src_ang[1],
                                            (pow(-1.0, k)) * src_ang[2]};
                                        double src_rec_vect[3] = {
                                            (-1) * R_p_plus_R_m[0],
                                            (-1) * R_p_plus_R_m[1],
                                            (-1) * R_p_plus_R_m[2]};

                                        if (sphRadius == 0) {
                                            const double directivity =
                                                src_directivity(look_dir_mir, src_rec_vect, src_type0);
                                            complex<double> reflbeta_scalar = 0;
                                            if (refl_coeff_ang_dep == 0) {
                                                reflbeta_scalar = directivity *
                                                    pow((complex<double>)beta[0], abs(mx - q)) * pow((complex<double>)beta[1], abs(mx)) *
                                                    pow((complex<double>)beta[2], abs(my - j)) * pow((complex<double>)beta[3], abs(my)) *
                                                    pow((complex<double>)beta[4], abs(mz - k)) * pow((complex<double>)beta[5], abs(mz));
                                            }
                                            for (int kk = 0; kk < k_total; kk++) {
                                                complex<double> Rb;
                                                if (refl_coeff_ang_dep == 0) {
                                                    Rb = reflbeta_scalar;
                                                } else {
                                                    for (int b = 0; b < 6; b++)
                                                        Q[b] = refl_factor_komatsu(refl_angles[b], N_FFT, fs, beta[b], kk);
                                                    Rb = directivity *
                                                        pow(Q[0], abs(mx - q)) * pow(Q[1], abs(mx)) *
                                                        pow(Q[2], abs(my - j)) * pow(Q[3], abs(my)) *
                                                        pow(Q[4], abs(mz - k)) * pow(Q[5], abs(mz));
                                                }
                                                complex<double> tmp_H =
                                                    Rb * exp(I * waveNr[kk] * R_p_plus_R_m_norm) / R_p_plus_R_m_norm;
                                                for (int ang = 0; ang < M; ang++) {
                                                    Hd[ang + M * kk] += tmp_H;
                                                }
                                            }
                                        } else {
                                            for (int ang = 0; ang < M; ang++) {
                                                tmp_angle = R_p_plus_R_m[0] * mic_pos[ang]
                                                          + R_p_plus_R_m[1] * mic_pos[ang + M]
                                                          + R_p_plus_R_m[2] * mic_pos[ang + 2 * M];
                                                if (tmp_angle < -1) tmp_angle = -1;
                                                else if (tmp_angle > 1) tmp_angle = 1;
                                                legendre(N_harm, tmp_angle, legendre_out);
                                                for (int ll = 0; ll <= N_harm; ll++)
                                                    shd_angle_l_dependent[ang + M * ll] =
                                                        legendre_out[ll] * shd_angle_l_dependent_all_sources[ll];
                                            }

                                            for (int kk = 0; kk < k_total; kk++) {
                                                int Nh = nhpk[kk]; if (Nh > N_harm) Nh = N_harm;
                                                sphbesselh(Nh, waveNr[kk] * R_p_plus_R_m_norm, sphbesselh_out);
                                                for (int ll = 0; ll <= Nh; ll++)
                                                    shd_k_l_dependent[kk + k_total * ll] =
                                                        sphbesselh_out[ll] * shd_k_l_p[kk + k_total * ll];
                                            }

                                            // Reflection*directivity factor is
                                            // independent of the microphone; for
                                            // real coefficients it is a single
                                            // scalar, otherwise one value per
                                            // frequency bin. Compute it once here
                                            // instead of M*k_total times below.
                                            const double directivity =
                                                src_directivity(look_dir_mir, src_rec_vect, src_type0);
                                            complex<double> reflbeta_scalar = 0;
                                            if (refl_coeff_ang_dep == 0) {
                                                reflbeta_scalar = directivity *
                                                    pow((complex<double>)beta[0], abs(mx - q)) * pow((complex<double>)beta[1], abs(mx)) *
                                                    pow((complex<double>)beta[2], abs(my - j)) * pow((complex<double>)beta[3], abs(my)) *
                                                    pow((complex<double>)beta[4], abs(mz - k)) * pow((complex<double>)beta[5], abs(mz));
                                            } else {
                                                for (int kk = 0; kk < k_total; kk++) {
                                                    for (int b = 0; b < 6; b++)
                                                        Q[b] = refl_factor_komatsu(refl_angles[b], N_FFT, fs, beta[b], kk);
                                                    reflbeta_k[kk] = directivity *
                                                        pow(Q[0], abs(mx - q)) * pow(Q[1], abs(mx)) *
                                                        pow(Q[2], abs(my - j)) * pow(Q[3], abs(my)) *
                                                        pow(Q[4], abs(mz - k)) * pow(Q[5], abs(mz));
                                                }
                                            }

                                            for (int ang = 0; ang < M; ang++) {
                                                for (int kk = 0; kk < k_total; kk++) {
                                                    int Nh = nhpk[kk]; if (Nh > N_harm) Nh = N_harm;
                                                    const complex<double> Rb =
                                                        (refl_coeff_ang_dep == 0) ? reflbeta_scalar : reflbeta_k[kk];
                                                    complex<double> tmp_H = 0;
                                                    for (int ll = 0; ll <= Nh; ll++)
                                                        tmp_H += shd_angle_l_dependent[ang + M * ll]
                                                               * shd_k_l_dependent[kk + k_total * ll];
                                                    Hd[ang + M * kk] += Rb * tmp_H;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // GIL re-acquired

    return H;
}

PYBIND11_MODULE(_smir_loop, m)
{
    m.doc() = "Native inner loop of the SMIR generator (image-source method).";
    m.def("smir_loop", &smir_loop,
          py::arg("c"), py::arg("fs"), py::arg("rr"), py::arg("ss"), py::arg("LL"),
          py::arg("beta"), py::arg("nsamples"), py::arg("order"), py::arg("K"),
          py::arg("shd_k_l"), py::arg("shd_angle_l"), py::arg("mic_pos"),
          py::arg("sphRadius"), py::arg("waveNr"), py::arg("refl_coeff_ang_dep"),
          py::arg("src_ang"), py::arg("src_type"), py::arg("n_harm_per_k"),
          "Compute the complex room transfer function H (M x k_total).");
}
