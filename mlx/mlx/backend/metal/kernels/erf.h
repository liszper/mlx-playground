#pragma once
#include <metal_math>

float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = 1.0f - metal::exp(r);
    r = metal::copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
  }
  return r;
}

float erfinv(float a) {
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float p;
  if (metal::abs(t) > 6.125f) {
    p = 3.03697567e-10f;
    p = metal::fma(p, t, 2.93243101e-8f);
    p = metal::fma(p, t, 1.22150334e-6f);
    p = metal::fma(p, t, 2.84108955e-5f);
    p = metal::fma(p, t, 3.93552968e-4f);
    p = metal::fma(p, t, 3.02698812e-3f);
    p = metal::fma(p, t, 4.83185798e-3f);
    p = metal::fma(p, t, -2.64646143e-1f);
    p = metal::fma(p, t, 8.40016484e-1f);
  } else {
    p = 5.43877832e-9f;
    p = metal::fma(p, t, 1.43285448e-7f);
    p = metal::fma(p, t, 1.22774793e-6f);
    p = metal::fma(p, t, 1.12963626e-7f);
    p = metal::fma(p, t, -5.61530760e-5f);
    p = metal::fma(p, t, -1.47697632e-4f);
    p = metal::fma(p, t, 2.31468678e-3f);
    p = metal::fma(p, t, 1.15392581e-2f);
    p = metal::fma(p, t, -2.32015476e-1f);
    p = metal::fma(p, t, 8.86226892e-1f);
  }
  return a * p;
}
