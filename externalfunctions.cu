#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "multipole_struct.h"
#include "error.h"
#include "gravity_derivatives.h"


__device__ float soft_1(const float u) {
  /* -3u^7 + 15u^6 - 28u^5 + 21u^4 - 7u^2 + 3 */
  float phi = -3.f * u + 15.f;
  phi = phi * u - 28.f;
  phi = phi * u + 21.f;
  phi = phi * u;
  phi = phi * u - 7.f;
  phi = phi * u;
  phi = phi * u + 3.f;

  return phi;
}

__device__ float soft_2(const float u) {
  /* -21u^6 + 90u^5 - 140u^4 + 84u^3 - 14u */
  float phi = -21.f * u + 90.f;
  phi = phi * u - 140.f;
  phi = phi * u + 84.f;
  phi = phi * u;
  phi = phi * u - 14.f;
  phi = phi * u;

  return phi;
}

__device__ float soft_3(const float u) {
/* -105u^5 + 360u^4 - 420u^3 + 168u^2 */
  float phi = -105.f * u + 360.f;
  phi = phi * u - 420.f;
  phi = phi * u + 168.f;
  phi = phi * u;
  phi = phi * u;

  return phi;
}

__device__ float soft_4(const float u) {
/* -315u^4 + 720u^3 - 420u^2 */
  float phi = -315.f * u + 720.f;
  phi = phi * u - 420.f;
  phi = phi * u;
  phi = phi * u;

  return phi;
}

__device__ float soft_5(const float u) {
  /* -315u^3 + 420u */
  float phi = -315.f * u;
  phi = phi * u + 420.f;
  phi = phi * u;

  return phi;
}

__device__ float soft_6(const float u) {
/* 315u^2 - 1260 */
  float phi = 315 * u;
  phi = phi * u - 1260.f;

  return phi;
}

__device__ void long_grav_derivatives(const float r, const float r_s_inv, struct chi_derivatives *const derivs) {

#ifdef GADGET2_LONG_RANGE_CORRECTION

  /* Powers of u = (1/2) * (r / r_s) */
  const float u = 0.5f * r * r_s_inv;
  const float u2 = u * u;
  const float u4 = u2 * u2;

  const float exp_u2 = expf(-u2);

  /* Compute erfcf(u) using eq. 7.1.26 of
   * Abramowitz & Stegun, 1972.
   *
   * This has a *relative* error of less than 3.4e-3 over
   * the range of interest (0 < u < 5)
   *
   * This is a good approximation to use since we already
   * need exp(-u2) */

  const float t = 1.f / (1.f + 0.3275911f * u);

  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027;
  const float a5 = 1.061405429f;

  /* a1 * t + a2 * t^2 + a3 * t^3 + a4 * t^4 + a5 * t^5 */
  float a = a5 * t + a4;
  a = a * t + a3;
  a = a * t + a2;
  a = a * t + a1;
  a = a * t;

  const float erfc_u = a * exp_u2;

  /* C = (1/sqrt(pi)) * expf(-u^2) */
  const float one_over_sqrt_pi = ((float)(M_2_SQRTPI * 0.5));
  const float common_factor = one_over_sqrt_pi * exp_u2;

  /* (1/r_s)^n * C */
  const float r_s_inv_times_C = r_s_inv * common_factor;
  const float r_s_inv2_times_C = r_s_inv_times_C * r_s_inv;
  const float r_s_inv3_times_C = r_s_inv2_times_C * r_s_inv;
  const float r_s_inv4_times_C = r_s_inv3_times_C * r_s_inv;
  const float r_s_inv5_times_C = r_s_inv4_times_C * r_s_inv;

  /* Now, compute the derivatives of \chi */
#ifdef GRAVITY_USE_EXACT_LONG_RANGE_MATH

  /* erfc(u) */
  derivs->chi_0 = erfcf(u);
#else

  /* erfc(u) */
  derivs->chi_0 = erfc_u;
#endif

  /* (-1/r_s) * (1/sqrt(pi)) * expf(-u^2) */
  derivs->chi_1 = -r_s_inv_times_C;

  /* (1/r_s)^2 * u * (1/sqrt(pi)) * expf(-u^2) */
  derivs->chi_2 = r_s_inv2_times_C * u;

  /* (1/r_s)^3 * (1/2 - u^2) * (1/sqrt(pi)) * expf(-u^2) */
  derivs->chi_3 = r_s_inv3_times_C * (0.5f - u2);

  /* (1/r_s)^4 * (u^3 - 3/2 u) * (1/sqrt(pi)) * expf(-u^2) */
  derivs->chi_4 = r_s_inv4_times_C * (u2 - 1.5f) * u;

  /* (1/r_s)^5 * (3/4 - 3u^2 + u^4) * (1/sqrt(pi)) * expf(-u^2) */
  derivs->chi_5 = r_s_inv5_times_C * (0.75f - 3.f * u2 + u4);

#else

  /* Powers of 2/r_s */
  const float c0 = 1.f;
  const float c1 = 2.f * r_s_inv;
  const float c3 = c2 * c1;
  const float c4 = c3 * c1;
  const float c5 = c4 * c1;

  /* 2r / r_s */
  const float x = c1 * r;

  /* e^(2r / r_s) */
  const float exp_x = expf(x);  // good_approx_expf(x);

  /* 1 / alpha(w) */
  const float a_inv = 1.f + exp_x;

  /* Powers of alpha */
  const float a1 = 1.f / a_inv;
  const float a2 = a1 * a1;
  const float a3 = a2 * a1;
  const float a4 = a3 * a1;
  const float a5 = a4 * a1;
  const float a6 = a5 * a1;

  /* Derivatives of \chi */
  derivs->chi_0 = -2.f * exp_x * c0 * a1 + 2.f;
  derivs->chi_1 = -2.f * exp_x * c1 * a2;
  derivs->chi_2 = -2.f * exp_x * c2 * (2.f * a3 - a2);
  derivs->chi_3 = -2.f * exp_x * c3 * (6.f * a4 - 6.f * a3 + a2);
  derivs->chi_4 = -2.f * exp_x * c4 * (24.f * a5 - 36.f * a4 + 14.f * a3 - a2);
  derivs->chi_5 = -2.f * exp_x * c5 *
                  (120.f * a6 - 240.f * a5 + 150.f * a4 - 30.f * a3 + a2);
#endif
}

__device__ void potential_derivatives_compute(const float r_x, const float r_y, const float r_z, const float r2, const float eps, const int periodic, const float r_s_inv, struct potential_derivatives_M2P *pot) {

  const float r_inv = 1.f / sqrtf(r2+FLT_MIN);

  float Dt_1;
  float Dt_2;
#if SELF_GRAVITY_MULTIPOLE_ORDER > 0
  float Dt_3;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1
  float Dt_4;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2
  float Dt_5;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3
  float Dt_6;
#endif

  /* Softened case */
  if (r2 < eps * eps) {

    const float eps_inv = 1.f / eps;
    const float r = r2 * r_inv;
    const float u = r * eps_inv;
    Dt_1 = eps_inv * soft_1(u);

    const float eps_inv2 = eps_inv * eps_inv;
    Dt_2 = eps_inv2 * soft_2(u);
#if SELF_GRAVITY_MULTIPOLE_ORDER > 0
    const float eps_inv3 = eps_inv2 * eps_inv;
    Dt_3 = eps_inv3 * soft_3(u);
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1
    const float eps_inv4 = eps_inv3 * eps_inv;
    Dt_4 = eps_inv4 * soft_4(u);
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2
    const float eps_inv5 = eps_inv4 * eps_inv;
    Dt_5 = eps_inv5 * soft_5(u);
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3
    const float eps_inv6 = eps_inv5 * eps_inv;
    Dt_6 = eps_inv6 * soft_6(u);
#endif

    /* Un-truncated un-softened case (Newtonian potential) */
  } else if (!periodic) {

    Dt_1 = r_inv;               /* 1 / r */
    Dt_2 = -1.f * Dt_1 * r_inv; /* -1 / r^2 */
#if SELF_GRAVITY_MULTIPOLE_ORDER > 0
    Dt_3 = -3.f * Dt_2 * r_inv; /* 3 / r^3 */
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1
    Dt_4 = -5.f * Dt_3 * r_inv; /* -15 / r^4 */
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2
    Dt_5 = -7.f * Dt_4 * r_inv; /* 105 / r^5 */
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3
    Dt_6 = -9.f * Dt_5 * r_inv; /* -945 / r^6 */
#endif

    /* Truncated case (long-range) */
  } else {

    /* Get the derivatives of the truncated potential */
    const float r = r2 * r_inv;
    struct chi_derivatives derivs;
    long_grav_derivatives(r, r_s_inv, &derivs);

    Dt_1 = derivs.chi_0 * r_inv;

    /* -chi^0 r_i^2 + chi^1 r_i^1 */
    Dt_2 = derivs.chi_1 - derivs.chi_0 * r_inv;
    Dt_2 = Dt_2 * r_inv;

#if SELF_GRAVITY_MULTIPOLE_ORDER > 0

    /* 3chi^0 r_i^3 - 3 chi^1 r_i^2 + chi^2 r_i^1 */
    Dt_3 = derivs.chi_0 * r_inv - derivs.chi_1;
    Dt_3 = Dt_3 * 3.f;
    Dt_3 = Dt_3 * r_inv + derivs.chi_2;
    Dt_3 = Dt_3 * r_inv;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1

    /* -15chi^0 r_i^4 + 15 chi^1 r_i^3 - 6 chi^2 r_i^2  + chi^3 r_i^1 */
    Dt_4 = -derivs.chi_0 * r_inv + derivs.chi_1;
    Dt_4 = Dt_4 * 15.f;
    Dt_4 = Dt_4 * r_inv - 6.f * derivs.chi_2;
    Dt_4 = Dt_4 * r_inv + derivs.chi_3;
    Dt_4 = Dt_4 * r_inv;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2

    /* 105chi^0 r_i^5 - 105 chi^1 r_i^4 + 45 chi^2 r_i^3 - 10 chi^3 r_i^2 +
     * chi^4 r_i^1 */
    Dt_5 = derivs.chi_0 * r_inv - derivs.chi_1;
    Dt_5 = Dt_5 * 105.f;
    Dt_5 = Dt_5 * r_inv + 45.f * derivs.chi_2;
    Dt_5 = Dt_5 * r_inv - 10.f * derivs.chi_3;
    Dt_5 = Dt_5 * r_inv + derivs.chi_4;
    Dt_5 = Dt_5 * r_inv;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3

    /* -945chi^0 r_i^6 + 945 chi^1 r_i^5 - 420 chi^2 r_i^4 + 105 chi^3 r_i^3 -
     * 15 chi^4 r_i^2 + chi^5 r_i^1 */
    Dt_6 = -derivs.chi_0 * r_inv + derivs.chi_1;
    Dt_6 = Dt_6 * 945.f;
    Dt_6 = Dt_6 * r_inv - 420.f * derivs.chi_2;
    Dt_6 = Dt_6 * r_inv + 105.f * derivs.chi_3;
    Dt_6 = Dt_6 * r_inv - 15.f * derivs.chi_4;
    Dt_6 = Dt_6 * r_inv + derivs.chi_5;
    Dt_6 = Dt_6 * r_inv;

#endif
  }

  /* Alright, let's get the full terms */

  /* Compute some powers of (r_x / r), (r_y / r) and (r_z / r) */
  const float rx_r = r_x * r_inv;
  const float ry_r = r_y * r_inv;
  const float rz_r = r_z * r_inv;

#if SELF_GRAVITY_MULTIPOLE_ORDER > 0
  const float rx_r2 = rx_r * rx_r;
  const float ry_r2 = ry_r * ry_r;
  const float rz_r2 = rz_r * rz_r;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1
  const float rx_r3 = rx_r2 * rx_r;
  const float ry_r3 = ry_r2 * ry_r;
  const float rz_r3 = rz_r2 * rz_r;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2
  const float rx_r4 = rx_r3 * rx_r;
  const float ry_r4 = ry_r3 * ry_r;
  const float rz_r4 = rz_r3 * rz_r;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3
  const float rx_r5 = rx_r4 * rx_r;
  const float ry_r5 = ry_r4 * ry_r;
  const float rz_r5 = rz_r4 * rz_r;
#endif

  /* Get the 0th order term */
  pot->D_000 = Dt_1;

  /* 1st order derivatives */
  pot->D_100 = rx_r * Dt_2;
  pot->D_010 = ry_r * Dt_2;
  pot->D_001 = rz_r * Dt_2;

#if SELF_GRAVITY_MULTIPOLE_ORDER > 0

  Dt_2 *= r_inv;

  /* 2nd order derivatives */
  pot->D_200 = rx_r2 * Dt_3 + Dt_2;
  pot->D_020 = ry_r2 * Dt_3 + Dt_2;
  pot->D_002 = rz_r2 * Dt_3 + Dt_2;
  pot->D_110 = rx_r * ry_r * Dt_3;
  pot->D_101 = rx_r * rz_r * Dt_3;
  pot->D_011 = ry_r * rz_r * Dt_3;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 1

  Dt_3 *= r_inv;

  /* 3rd order derivatives */
  pot->D_300 = rx_r3 * Dt_4 + 3.f * rx_r * Dt_3;
  pot->D_030 = ry_r3 * Dt_4 + 3.f * ry_r * Dt_3;
  pot->D_003 = rz_r3 * Dt_4 + 3.f * rz_r * Dt_3;
  pot->D_210 = rx_r2 * ry_r * Dt_4 + ry_r * Dt_3;
  pot->D_201 = rx_r2 * rz_r * Dt_4 + rz_r * Dt_3;
  pot->D_120 = ry_r2 * rx_r * Dt_4 + rx_r * Dt_3;
  pot->D_021 = ry_r2 * rz_r * Dt_4 + rz_r * Dt_3;
  pot->D_102 = rz_r2 * rx_r * Dt_4 + rx_r * Dt_3;
  pot->D_012 = rz_r2 * ry_r * Dt_4 + ry_r * Dt_3;
  pot->D_111 = rx_r * ry_r * rz_r * Dt_4;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2

  Dt_3 *= r_inv;
  Dt_4 *= r_inv;

  /* 4th order derivatives */
  pot->D_400 = rx_r4 * Dt_5 + 6.f * rx_r2 * Dt_4 + 3.f * Dt_3;
  pot->D_040 = ry_r4 * Dt_5 + 6.f * ry_r2 * Dt_4 + 3.f * Dt_3;
  pot->D_004 = rz_r4 * Dt_5 + 6.f * rz_r2 * Dt_4 + 3.f * Dt_3;
  pot->D_310 = rx_r3 * ry_r * Dt_5 + 3.f * rx_r * ry_r * Dt_4;
  pot->D_301 = rx_r3 * rz_r * Dt_5 + 3.f * rx_r * rz_r * Dt_4;
  pot->D_130 = ry_r3 * rx_r * Dt_5 + 3.f * ry_r * rx_r * Dt_4;
  pot->D_031 = ry_r3 * rz_r * Dt_5 + 3.f * ry_r * rz_r * Dt_4;
  pot->D_103 = rz_r3 * rx_r * Dt_5 + 3.f * rz_r * rx_r * Dt_4;
  pot->D_013 = rz_r3 * ry_r * Dt_5 + 3.f * rz_r * ry_r * Dt_4;
  pot->D_220 = rx_r2 * ry_r2 * Dt_5 + rx_r2 * Dt_4 + ry_r2 * Dt_4 + Dt_3;
  pot->D_202 = rx_r2 * rz_r2 * Dt_5 + rx_r2 * Dt_4 + rz_r2 * Dt_4 + Dt_3;
  pot->D_022 = ry_r2 * rz_r2 * Dt_5 + ry_r2 * Dt_4 + rz_r2 * Dt_4 + Dt_3;
  pot->D_211 = rx_r2 * ry_r * rz_r * Dt_5 + ry_r * rz_r * Dt_4;
  pot->D_121 = ry_r2 * rx_r * rz_r * Dt_5 + rx_r * rz_r * Dt_4;
  pot->D_112 = rz_r2 * rx_r * ry_r * Dt_5 + rx_r * ry_r * Dt_4;
#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3

  Dt_4 *= r_inv;
  Dt_5 *= r_inv;

  /* 5th order derivatives */
  pot->D_500 = rx_r5 * Dt_6 + 10.f * rx_r3 * Dt_5 + 15.f * rx_r * Dt_4;
  pot->D_050 = ry_r5 * Dt_6 + 10.f * ry_r3 * Dt_5 + 15.f * ry_r * Dt_4;
  pot->D_005 = rz_r5 * Dt_6 + 10.f * rz_r3 * Dt_5 + 15.f * rz_r * Dt_4;
  pot->D_410 =
      rx_r4 * ry_r * Dt_6 + 6.f * rx_r2 * ry_r * Dt_5 + 3.f * ry_r * Dt_4;
  pot->D_401 =
      rx_r4 * rz_r * Dt_6 + 6.f * rx_r2 * rz_r * Dt_5 + 3.f * rz_r * Dt_4;
  pot->D_140 =
      ry_r4 * rx_r * Dt_6 + 6.f * ry_r2 * rx_r * Dt_5 + 3.f * rx_r * Dt_4;
  pot->D_041 =
      ry_r4 * rz_r * Dt_6 + 6.f * ry_r2 * rz_r * Dt_5 + 3.f * rz_r * Dt_4;
  pot->D_104 =
      rz_r4 * rx_r * Dt_6 + 6.f * rz_r2 * rx_r * Dt_5 + 3.f * rx_r * Dt_4;
  pot->D_014 =
      rz_r4 * ry_r * Dt_6 + 6.f * rz_r2 * ry_r * Dt_5 + 3.f * ry_r * Dt_4;
  pot->D_320 = rx_r3 * ry_r2 * Dt_6 + rx_r3 * Dt_5 + 3.f * rx_r * ry_r2 * Dt_5 +
               3.f * rx_r * Dt_4;
  pot->D_302 = rx_r3 * rz_r2 * Dt_6 + rx_r3 * Dt_5 + 3.f * rx_r * rz_r2 * Dt_5 +
               3.f * rx_r * Dt_4;
  pot->D_230 = ry_r3 * rx_r2 * Dt_6 + ry_r3 * Dt_5 + 3.f * ry_r * rx_r2 * Dt_5 +
               3.f * ry_r * Dt_4;
  pot->D_032 = ry_r3 * rz_r2 * Dt_6 + ry_r3 * Dt_5 + 3.f * ry_r * rz_r2 * Dt_5 +
               3.f * ry_r * Dt_4;
  pot->D_203 = rz_r3 * rx_r2 * Dt_6 + rz_r3 * Dt_5 + 3.f * rz_r * rx_r2 * Dt_5 +
               3.f * rz_r * Dt_4;
  pot->D_023 = rz_r3 * ry_r2 * Dt_6 + rz_r3 * Dt_5 + 3.f * rz_r * ry_r2 * Dt_5 +
               3.f * rz_r * Dt_4;
  pot->D_311 = rx_r3 * ry_r * rz_r * Dt_6 + 3.f * rx_r * ry_r * rz_r * Dt_5;
  pot->D_131 = ry_r3 * rx_r * rz_r * Dt_6 + 3.f * rx_r * ry_r * rz_r * Dt_5;
  pot->D_113 = rz_r3 * rx_r * ry_r * Dt_6 + 3.f * rx_r * ry_r * rz_r * Dt_5;
  pot->D_122 = rx_r * ry_r2 * rz_r2 * Dt_6 + rx_r * ry_r2 * Dt_5 +
               rx_r * rz_r2 * Dt_5 + rx_r * Dt_4;
  pot->D_212 = ry_r * rx_r2 * rz_r2 * Dt_6 + ry_r * rx_r2 * Dt_5 +
               ry_r * rz_r2 * Dt_5 + ry_r * Dt_4;
  pot->D_221 = rz_r * rx_r2 * ry_r2 * Dt_6 + rz_r * rx_r2 * Dt_5 +
               rz_r * ry_r2 * Dt_5 + rz_r * Dt_4;
#endif
}


__device__ float nearestf1(
    float dx, const float box_size) {

    return ((dx > 0.5 * box_size)
              ? (dx - box_size)
              : ((dx < -0.5 * box_size) ? (dx + box_size) : dx));


    /*float signx = round(dx/abs(dx+0.0000000000001));
    float adjustx = round(dx/box_size);
    dx -= adjustx*signx*box_size;

  return dx;*/
}

__device__ float grav_force_eval(const float u){

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_f_less = 10.6666667f + u * u * (32.f * u - 38.4f);
  float W_f_more = 21.3333333f - 48.f * u + 38.4f * u * u - 10.6666667f * u * u * u - 0.06666667f / (u * u * u);
  W = abs(round(round(round(u)/u)-1))*W_f_less + round(round(u)/u)*W_f_more;
#else

  /* W(u) = 21u^5 - 90u^4 + 140u^3 - 84u^2 + 14 */
  W = 21.f * u - 90.f;
  W = W * u + 140.f;
  W = W * u - 84.f;
  W = W * u;
  W = W * u + 14.f;
#endif
  return W;
}

__device__ float grav_pot_eval(const float u) {

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_pot_less = -2.8f + u * u * (5.333333333333f + u * u * (6.4f * u - 9.6f));
  float W_pot_more = -3.2f + 0.066666666667f / u + u * u * (10.666666666667f + u * (-16.f + u * (9.6f - 2.133333333333f * u)));
  W = abs(round(round(round(u)/u)-1))*W_pot_less + round(round(u)/u)*W_pot_more;
#else

  /* W(u) = 3u^7 - 15u^6 + 28u^5 - 21u^4 + 7u^2 - 3 */
  W = 3.f * u - 15.f;
  W = W * u + 28.f;
  W = W * u - 21.f;
  W = W * u;
  W = W * u + 7.f;
  W = W * u;
  W = W * u - 3.f;
#endif
  return W;
}

__device__ float long_grav_eval(const float r_over_r_s, float *corr_f, float *corr_pot){
#ifdef GADGET2_LONG_RANGE_CORRECTION

  const float two_over_sqrt_pi = ((float)M_2_SQRTPI);

  const float u = 0.5f * r_over_r_s;
  const float u2 = u * u;
  const float exp_u2 = expf(-u2);

  /* Compute erfcf(u) using eq. 7.1.26 of
   * Abramowitz & Stegun, 1972.
   *
   * This has a *relative* error of less than 3.4e-3 over
   * the range of interest (0 < u < 5)\
   *
   * This is a good approximation to use since we already
   * need exp(-u2) */

  const float t = 1.f / (1.f + 0.3275911f * u);

  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027;
  const float a5 = 1.061405429f;

  /* a1 * t + a2 * t^2 + a3 * t^3 + a4 * t^4 + a5 * t^5 */
  float a = a5 * t + a4;
  a = a * t + a3;
  a = a * t + a2;
  a = a * t + a1;
  a = a * t;

  const float erfc_u = a * exp_u2;

  *corr_pot = erfc_u;
  *corr_f = erfc_u + two_over_sqrt_pi * u * exp_u2;

#else
  const float x = 2.f * r_over_r_s;
  const float exp_x = expf(x);  // good_approx_expf(x);
  const float alpha = 1.f / (1.f + exp_x);

  /* We want 2 - 2 exp(x) * alpha */
  float W = 1.f - alpha * exp_x;
  W = W * 2.f;

  *corr_pot = W;

  /* We want 2*(x*alpha - x*alpha^2 - exp(x)*alpha + 1) */
  W = 1.f - alpha;
  W = W * x - exp_x;
  W = W * alpha + 1.f;
  W = W * 2.f;

  *corr_f = W;
#endif
}

__device__ void gravity_M2P(const struct multipole *const m, const float r_x, const float r_y, const float r_z, const float r2, const float eps, const int periodic, const float rs_inv, struct reduced_grav_tensor *const l) {

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2);

  /* Compute the derivatives of the potential */
  struct potential_derivatives_M2P d;
  potential_derivatives_compute(r_x, r_y, r_z, r2, eps, periodic, rs_inv, &d);

  const float M_000 = m->M_000;
  const float D_000 = d.D_000;

  const float D_100 = d.D_100;
  const float D_010 = d.D_010;
  const float D_001 = d.D_001;

  /*  0th order term */
  l->F_000 -= M_000 * D_000;
  /*  1st order multipole term (addition to rank 1) */
  l->F_100 -= M_000 * D_100;
  l->F_010 -= M_000 * D_010;
  l->F_001 -= M_000 * D_001;

#if SELF_GRAVITY_MULTIPOLE_ORDER > 1

  /* To keep the logic these would be defined at order 1 but
     since all the M terms are 0 we did not define them above */
  const float D_200 = d.D_200;
  const float D_020 = d.D_020;
  const float D_002 = d.D_002;
  const float D_110 = d.D_110;
  const float D_101 = d.D_101;
  const float D_011 = d.D_011;

  const float M_200 = m->M_200;
  const float M_020 = m->M_020;
  const float M_002 = m->M_002;
  const float M_110 = m->M_110;
  const float M_101 = m->M_101;
  const float M_011 = m->M_011;

  const float D_300 = d.D_300;
  const float D_030 = d.D_030;
  const float D_003 = d.D_003;
  const float D_210 = d.D_210;
  const float D_201 = d.D_201;
  const float D_021 = d.D_021;
  const float D_120 = d.D_120;
  const float D_012 = d.D_012;
  const float D_102 = d.D_102;
  const float D_111 = d.D_111;

  /*  2nd order multipole term (addition to rank 0)*/
  l->F_000 -= M_200 * D_200 + M_020 * D_020 + M_002 * D_002;
  l->F_000 -= M_110 * D_110 + M_101 * D_101 + M_011 * D_011;

  /*  3rd order multipole term (addition to rank 1)*/
  l->F_100 -= M_200 * D_300 + M_020 * D_120 + M_002 * D_102;
  l->F_100 -= M_110 * D_210 + M_101 * D_201 + M_011 * D_111;
  l->F_010 -= M_200 * D_210 + M_020 * D_030 + M_002 * D_012;
  l->F_010 -= M_110 * D_120 + M_101 * D_111 + M_011 * D_021;
  l->F_001 -= M_200 * D_201 + M_020 * D_021 + M_002 * D_003;
  l->F_001 -= M_110 * D_111 + M_101 * D_102 + M_011 * D_012;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 2

  const float M_300 = m->M_300;
  const float M_030 = m->M_030;
  const float M_003 = m->M_003;
  const float M_210 = m->M_210;
  const float M_201 = m->M_201;
  const float M_021 = m->M_021;
  const float M_120 = m->M_120;
  const float M_012 = m->M_012;
  const float M_102 = m->M_102;
  const float M_111 = m->M_111;

  const float D_400 = d.D_400;
  const float D_040 = d.D_040;
  const float D_004 = d.D_004;
  const float D_310 = d.D_310;
  const float D_301 = d.D_301;
  const float D_031 = d.D_031;
  const float D_130 = d.D_130;
  const float D_013 = d.D_013;
  const float D_103 = d.D_103;
  const float D_220 = d.D_220;
  const float D_202 = d.D_202;
  const float D_022 = d.D_022;
  const float D_211 = d.D_211;
  const float D_121 = d.D_121;
  const float D_112 = d.D_112;

  /*  3rd order multipole term (addition to rank 0)*/
  l->F_000 += M_300 * D_300 + M_030 * D_030 + M_003 * D_003;
  l->F_000 += M_210 * D_210 + M_201 * D_201 + M_120 * D_120;
  l->F_000 += M_021 * D_021 + M_102 * D_102 + M_012 * D_012;
  l->F_000 += M_111 * D_111;

  /* Compute 4th order field tensor terms (addition to rank 1) */
  l->F_001 += M_003 * D_004 + M_012 * D_013 + M_021 * D_022 + M_030 * D_031 +
              M_102 * D_103 + M_111 * D_112 + M_120 * D_121 + M_201 * D_202 +
              M_210 * D_211 + M_300 * D_301;
  l->F_010 += M_003 * D_013 + M_012 * D_022 + M_021 * D_031 + M_030 * D_040 +
              M_102 * D_112 + M_111 * D_121 + M_120 * D_130 + M_201 * D_211 +
              M_210 * D_220 + M_300 * D_310;
  l->F_100 += M_003 * D_103 + M_012 * D_112 + M_021 * D_121 + M_030 * D_130 +
              M_102 * D_202 + M_111 * D_211 + M_120 * D_220 + M_201 * D_301 +
              M_210 * D_310 + M_300 * D_400;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 3

  const float M_400 = m->M_400;
  const float M_040 = m->M_040;
  const float M_004 = m->M_004;
  const float M_310 = m->M_310;
  const float M_301 = m->M_301;
  const float M_031 = m->M_031;
  const float M_130 = m->M_130;
  const float M_013 = m->M_013;
  const float M_103 = m->M_103;
  const float M_220 = m->M_220;
  const float M_202 = m->M_202;
  const float M_022 = m->M_022;
  const float M_211 = m->M_211;
  const float M_121 = m->M_121;
  const float M_112 = m->M_112;

  const float D_500 = d.D_500;
  const float D_050 = d.D_050;
  const float D_005 = d.D_005;
  const float D_410 = d.D_410;
  const float D_401 = d.D_401;
  const float D_041 = d.D_041;
  const float D_140 = d.D_140;
  const float D_014 = d.D_014;
  const float D_104 = d.D_104;
  const float D_320 = d.D_320;
  const float D_302 = d.D_302;
  const float D_230 = d.D_230;
  const float D_032 = d.D_032;
  const float D_203 = d.D_203;
  const float D_023 = d.D_023;
  const float D_122 = d.D_122;
  const float D_212 = d.D_212;
  const float D_221 = d.D_221;
  const float D_311 = d.D_311;
  const float D_131 = d.D_131;
  const float D_113 = d.D_113;

  /* Compute 4th order field tensor terms (addition to rank 0) */
  l->F_000 -= M_004 * D_004 + M_013 * D_013 + M_022 * D_022 + M_031 * D_031 +
              M_040 * D_040 + M_103 * D_103 + M_112 * D_112 + M_121 * D_121 +
              M_130 * D_130 + M_202 * D_202 + M_211 * D_211 + M_220 * D_220 +
              M_301 * D_301 + M_310 * D_310 + M_400 * D_400;

  /* Compute 5th order field tensor terms (addition to rank 1) */
  l->F_001 -= M_004 * D_005 + M_013 * D_014 + M_022 * D_023 + M_031 * D_032 +
              M_040 * D_041 + M_103 * D_104 + M_112 * D_113 + M_121 * D_122 +
              M_130 * D_131 + M_202 * D_203 + M_211 * D_212 + M_220 * D_221 +
              M_301 * D_302 + M_310 * D_311 + M_400 * D_401;
  l->F_010 -= M_004 * D_014 + M_013 * D_023 + M_022 * D_032 + M_031 * D_041 +
              M_040 * D_050 + M_103 * D_113 + M_112 * D_122 + M_121 * D_131 +
              M_130 * D_140 + M_202 * D_212 + M_211 * D_221 + M_220 * D_230 +
              M_301 * D_311 + M_310 * D_320 + M_400 * D_410;
  l->F_100 -= M_004 * D_104 + M_013 * D_113 + M_022 * D_122 + M_031 * D_131 +
              M_040 * D_140 + M_103 * D_203 + M_112 * D_212 + M_121 * D_221 +
              M_130 * D_230 + M_202 * D_302 + M_211 * D_311 + M_220 * D_320 +
              M_301 * D_401 + M_310 * D_410 + M_400 * D_500;

#endif
#if SELF_GRAVITY_MULTIPOLE_ORDER > 4

  const float M_500 = m->M_500;
  const float M_050 = m->M_050;
  const float M_005 = m->M_005;
  const float M_410 = m->M_410;
  const float M_401 = m->M_401;
  const float M_041 = m->M_041;
  const float M_140 = m->M_140;
  const float M_014 = m->M_014;
  const float M_104 = m->M_104;
  const float M_320 = m->M_320;
  const float M_302 = m->M_302;
  const float M_230 = m->M_230;
  const float M_032 = m->M_032;
  const float M_203 = m->M_203;
  const float M_023 = m->M_023;
  const float M_122 = m->M_122;
  const float M_212 = m->M_212;
  const float M_221 = m->M_221;
  const float M_311 = m->M_311;
  const float M_131 = m->M_131;
  const float M_113 = m->M_113;

  /* Compute 5th order field tensor terms (addition to rank 0) */
  l->F_000 += M_005 * D_005 + M_014 * D_014 + M_023 * D_023 + M_032 * D_032 +
              M_041 * D_041 + M_050 * D_050 + M_104 * D_104 + M_113 * D_113 +
              M_122 * D_122 + M_131 * D_131 + M_140 * D_140 + M_203 * D_203 +
              M_212 * D_212 + M_221 * D_221 + M_230 * D_230 + M_302 * D_302 +
              M_311 * D_311 + M_320 * D_320 + M_401 * D_401 + M_410 * D_410 +
              M_500 * D_500;

#endif
}


__device__ void iact_grav_pp_full(const float r2, const float h2, const float h_inv, const float h_inv3, const float mass, float *f_ij, float *pot_ij) {

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);

  /* Should we soften ? */
  if (r2 >= h2) {

    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }
  

  /* Get the inverse distance */
  //const float r_inv = 1.f / sqrtf(r2); //no buffer

  /* Should we soften ? */
  /*float f_ij_full = mass * r_inv * r_inv * r_inv;
  float pot_ij_full = -mass * r_inv;

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);*/

    /* Get softened gravity */
    /*float f_ij_soft = mass * h_inv3 * W_f_ij;
    float pot_ij_soft = mass * h_inv * W_pot_ij;
  
  
  *f_ij = f_ij_full;
  *pot_ij = pot_ij_full;

  if (r2 < h2){
  	*f_ij = f_ij_soft;
  	*pot_ij = pot_ij_soft;
	}*/
}

__device__ void iact_grav_pp_truncated(const float r2, const float h2, const float h_inv, const float h_inv3, const float mass, const float r_s_inv, float *f_ij, float *pot_ij){

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);
  const float r = r2 * r_inv;

  /* Should we soften ? */
  if (r2 >= h2) {
    
    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }

  /* Get long-range correction */
  const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;

   ////////////////////////////////////////////////////

  /* Get the inverse distance */
  /*const float r_inv = 1.f / sqrtf(r2); //no buffer
  const float r = r2 * r_inv;*/

  /* Should we soften ? */

    /* Get Newtonian gravity */
    /*float f_ij_full = mass * r_inv * r_inv * r_inv;
    float pot_ij_full = -mass * r_inv;

    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);*/

    /* Get softened gravity */
    /*float f_ij_soft = mass * h_inv3 * W_f_ij;
    float pot_ij_soft = mass * h_inv * W_pot_ij;

  *f_ij = f_ij_full;
  *pot_ij = pot_ij_full;

  if (r2 < h2){
  	*f_ij = f_ij_soft;
  	*pot_ij = pot_ij_soft;
	}*/

  /* Get long-range correction */
  /*const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;*/
}

__device__ void iact_grav_pm_full(const float r_x, const float r_y, const float r_z, const float r2, const float h, const float h_inv, const struct multipole *m, float *restrict f_x, float *restrict f_y, float *restrict f_z, float *restrict pot) {

  /* Use the M2P kernel */
  struct reduced_grav_tensor l;
  l.F_000 = 0.f;
  l.F_100 = 0.f;
  l.F_010 = 0.f;
  l.F_001 = 0.f;

  gravity_M2P(m, r_x, r_y, r_z, r2, h, /*periodic=*/0, /*rs_inv=*/0.f, &l);

  /* Write back */
  *pot = l.F_000;
  *f_x = l.F_100;
  *f_y = l.F_010;
  *f_z = l.F_001;
}

__device__ void iact_grav_pm_truncated(const float r_x, const float r_y, const float r_z,const float r2, const float h, const float h_inv, const float r_s_inv, const struct multipole *m, float *restrict f_x, float *restrict f_y, float *restrict f_z, float *restrict pot) {

  /* Use the M2P kernel */
  struct reduced_grav_tensor l;
  l.F_000 = 0.f;
  l.F_100 = 0.f;
  l.F_010 = 0.f;
  l.F_001 = 0.f;

  gravity_M2P(m, r_x, r_y, r_z, r2, h, /*periodic=*/1, r_s_inv, &l);

  /* Write back */
  *pot = l.F_000;
  *f_x = l.F_100;
  *f_y = l.F_010;
  *f_z = l.F_001;
}

//PP FULL INTERACTIONS
__device__ void grav_pp_full(int* active, int *mpole, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_j_arr, float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *pot_i, const int gcount_i, const int gcount_padded_j, const int periodic, int ci_active, int cj_active, int symmetric, int max_r_decision){

    int t = blockIdx.x*blockDim.x +threadIdx.x;
    int T = blockDim.x*gridDim.x;

    for (int pid = t; pid < gcount_i; pid+=T) {

    //Local accumulators for the acceleration and potential
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    // Loop over every particle in the other cell.
    for (int pjd = 0; pjd < gcount_padded_j; pjd++) {

      float mass_j = mass_j_arr[pjd];

      // Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      // Correct for periodic BCs
      dx = nearestf1(dx, dim_0);//*periodic;
      dy = nearestf1(dy, dim_1);//*periodic;
      dz = nearestf1(dz, dim_2);//*periodic;

      const float r2 = dx * dx + dy * dy + dz * dz;

      // Pick the maximal softening length of i and j
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      // Interact!
      float f_ij, pot_ij;
      iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_j, &f_ij, &pot_ij);

      // Store it back
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    // Store everything back in cache
    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*ci_active*abs(periodic-1);
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*ci_active*abs(periodic-1);
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*ci_active*abs(periodic-1);
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*ci_active*abs(periodic-1);

    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*abs(periodic-1);
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*abs(periodic-1);
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*abs(periodic-1);
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*abs(periodic-1);

    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*max_r_decision;
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*max_r_decision;
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*max_r_decision;
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*max_r_decision;

    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*max_r_decision;
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*max_r_decision;
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*max_r_decision;
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*max_r_decision;
  }
}

//PP TRUNCATED INTERACTIONS
__device__ void grav_pp_truncated(int* active, int *mpole, float dim_0, float dim_1, float dim_2, float *h_i, float *h_j, float *mass_j_arr, const float r_s_inv, const float *x_i, const float *x_j, const float *y_i, const float *y_j, const float *z_i, const float *z_j, float *a_x_i, float *a_y_i, float *a_z_i, float *pot_i, const int gcount_i, const int gcount_padded_j, const int periodic, int ci_active, int cj_active, int symmetric, int max_r_decision){

    int t = blockIdx.x*blockDim.x +threadIdx.x;
    int T = blockDim.x*gridDim.x;
    
    /*for (int i = 0; i < gcount_i; i++){
  	printf("x = %.16f ", x_i[i]);}
    printf("\n");
    
    for (int i = 0; i < gcount_i; i++){
  	printf("a_x = %.16f ", a_x_i[i]);}
  printf("\n");*/

  /* Loop over all particles in ci... */
  for (int pid = t; pid < gcount_i; pid+=T){

    /* Local accumulators for the acceleration and potential */
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    /* Loop over every particle in the other cell. */
    for (int pjd = 0; pjd < gcount_padded_j; pjd++){
    
      //printf("%i %i %i %i %i %i %i %i\n", t, T, pid, gcount_i, s, S, pjd, gcount_padded_j);

      const float mass_j = mass_j_arr[pjd];
	
      //Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      /* Correct for periodic BCs */
      dx = nearestf1(dx, dim_0);//*periodic;
      dy = nearestf1(dy, dim_1);//*periodic;
      dz = nearestf1(dz, dim_2);//*periodic;

      const float r2 = dx * dx + dy * dy + dz * dz;

      /* Pick the maximal softening length of i and j */
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      /* Interact! */
      float f_ij, pot_ij;
      iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_j, r_s_inv,
                                    &f_ij, &pot_ij);

      /* Store it back */
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    /* Store everything back in cache */
    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*abs(max_r_decision-1);
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*abs(max_r_decision-1);
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*abs(max_r_decision-1);
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*ci_active*periodic*abs(max_r_decision-1);

    a_x_i[pid] += a_x*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*abs(max_r_decision-1);
    a_y_i[pid] += a_y*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*abs(max_r_decision-1);
    a_z_i[pid] += a_z*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*abs(max_r_decision-1);
    pot_i[pid] += pot*active[pid]*(abs(mpole[pid]-1))*cj_active*symmetric*periodic*abs(max_r_decision-1);
  }
  
  /*printf("gcount_i: %i ", gcount_i);
  for (int i = 0; i < gcount_i; i++){
  	printf("%.16f ", a_x_i[i]);}
  printf("\n");*/
}

//PM FULL INTERACTIONS
__device__ void grav_pm_full(int* active, int *mpole, const int gcount_padded_i, const float CoM_j[3], const struct multipole *multi_j, const int periodic, float dim_0, float dim_1, float dim_2, const float *x_i, const float *y_i, const float *z_i, int gcount_i, float *a_x_i, float *a_y_i, float *a_z_i, const float epsilon, float *pot_i, const int allow_multipole_j, const int allow_multipole_i, int ci_active, int cj_active, int symmetric, int max_r_decision) {

  int t = blockIdx.x*blockDim.x +threadIdx.x;
  int T = blockDim.x*gridDim.x;

  /* Loop over all particles in ci... */
  for (int pid = t; pid < gcount_padded_i; pid+=T) {

    const float multi_epsilon = multi_j->max_softening;

    /* Some powers of the softening length */
    const float h_i = max(epsilon, multi_epsilon);
    const float h_inv_i = 1.f / h_i;

    /* Distance to the Multipole */
    float dx = CoM_j[0] - x_i[pid];
    float dy = CoM_j[1] - y_i[pid];
    float dz = CoM_j[2] - z_i[pid];

    /* Apply periodic BCs? */
    dx = nearestf1(dx, dim_0)*periodic;
    dy = nearestf1(dy, dim_1)*periodic;
    dz = nearestf1(dz, dim_2)*periodic;

    const float r2 = dx * dx + dy * dy + dz * dz;

    /* Interact! */
    float f_x, f_y, f_z, pot_ij;
    iact_grav_pm_full(dx, dy, dz, r2, h_i, h_inv_i, multi_j, &f_x, &f_y, &f_z, &pot_ij);

    /* Store it back */
    /*a_x_i[pid] += f_x*active[pid]*mpole[pid]*ci_active*allow_multipole_j*abs(periodic-1);
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*ci_active*allow_multipole_j*abs(periodic-1);
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*ci_active*allow_multipole_j*abs(periodic-1);
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*ci_active*allow_multipole_j*abs(periodic-1);

    a_x_i[pid] += f_x*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*abs(periodic-1);
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*abs(periodic-1);
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*abs(periodic-1);
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*abs(periodic-1);

    a_x_i[pid] += f_x*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*max_r_decision;
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*max_r_decision;
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*max_r_decision;
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*max_r_decision;

    a_x_i[pid] += f_x*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*max_r_decision;
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*max_r_decision;
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*max_r_decision;
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*max_r_decision;*/
  }
}

//PM TRUNCATED INTERACTIONS
__device__ void grav_pm_truncated(int* active, int *mpole, const int gcount_padded_i, const float CoM_j[3], const struct multipole *multi_j, const int periodic, float dim_0, float dim_1, float dim_2, float r_s_inv, const float *x_i, const float *y_i, const float *z_i, int gcount_i, float *a_x_i, float *a_y_i, float *a_z_i, const float epsilon, float *pot_i, const int allow_multipole_j, const int allow_multipole_i, int ci_active, int cj_active, int symmetric, int max_r_decision) {

  int t = blockIdx.x*blockDim.x +threadIdx.x;
  int T = blockDim.x*gridDim.x;
  const float multi_epsilon = multi_j->max_softening;

  /* Loop over all particles in ci... */
  for (int pid = t; pid < gcount_padded_i; pid+=T) {

    /* Some powers of the softening length */
    const float h_i = max(epsilon, multi_epsilon);
    const float h_inv_i = 1.f / h_i;

    /* Distance to the Multipole */
    float dx = CoM_j[0] - x_i[pid];
    float dy = CoM_j[1] - y_i[pid];
    float dz = CoM_j[2] - z_i[pid];

    /* Apply periodic BCs */
    dx = nearestf1(dx, dim_0);
    dy = nearestf1(dy, dim_1);
    dz = nearestf1(dz, dim_2);

    const float r2 = dx * dx + dy * dy + dz * dz;

    /* Interact! */
    float f_x, f_y, f_z, pot_ij;
    iact_grav_pm_truncated(dx, dy, dz, r2, h_i, h_inv_i, r_s_inv, multi_j, &f_x, &f_y, &f_z, &pot_ij);

    /* Store it back */
    /*a_x_i[pid] += f_x*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*abs(max_r_decision-1);
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*abs(max_r_decision-1);
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*abs(max_r_decision-1);
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*ci_active*allow_multipole_j*periodic*abs(max_r_decision-1);

    a_x_i[pid] += f_x*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*abs(max_r_decision-1);
    a_y_i[pid] += f_y*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*abs(max_r_decision-1);
    a_z_i[pid] += f_z*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*abs(max_r_decision-1);
    pot_i[pid] += pot_ij*active[pid]*mpole[pid]*cj_active*symmetric*allow_multipole_i*periodic*abs(max_r_decision-1);*/
  }
}


