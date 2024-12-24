#pragma once

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/erf.h"
#include "mlx/backend/metal/kernels/expm1f.h"

namespace {
constant float inf = metal::numeric_limits<float>::infinity();
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return metal::abs(x);
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return metal::ceil(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cosh(x);
  };
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return metal::precise::exp(x);
  };
};

struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(expm1f(static_cast<float>(x)));
  };
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return metal::floor(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log(x);
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log2(x);
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log10(x);
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return metal::rint(x);
  };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sin(x);
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sinh(x);
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sqrt(x);
  };
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tanh(x);
  };
};
