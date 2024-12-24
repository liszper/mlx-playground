#pragma once

#include <metal_math>

/* Compute exponential base e minus 1. Maximum ulp error = 0.997458

   i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
   Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
   With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
   when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

   NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
*/
float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;

  j = fma(1.442695f, a, 12582912.f);
  j = j - 12582912.0f;
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);

  s = f * f;
  if (a == 0.0f)
    s = a;
  r = 1.97350979e-4f;
  r = fma(r, f, 1.39309070e-3f);
  r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f);
  r = fma(r, f, 1.66666716e-1f);
  r = fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}

float expm1f(float a) {
  float r;

  r = expm1f_scaled_unchecked(a, 1.0f);
  /* handle severe overflow and underflow */
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}
