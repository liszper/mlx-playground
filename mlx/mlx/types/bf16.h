#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define __MLX_BFLOAT_NAN__ 0x7FC0

namespace mlx::core {

namespace {
union float_bits_bf16 {
  float f;
  uint32_t u;
};
}

struct _MLX_BFloat16 {
  uint16_t bits_;

  _MLX_BFloat16() = default;

  _MLX_BFloat16(_MLX_BFloat16 const&) = default;

  _MLX_BFloat16& operator=(std::vector<bool>::reference x) {
    bits_ = x;
    return *this;
  }

  _MLX_BFloat16& operator=(const float& x) {
    return (*this = _MLX_BFloat16(x));
  }

  _MLX_BFloat16(const float& x) {
    if (std::isnan(x)) {
      bits_ = __MLX_BFLOAT_NAN__;
    } else {
      float_bits_bf16 in;

      in.f = x;

      in.u += (in.u >> 16 & 1) + uint32_t(0x7FFF);

      bits_ = in.u >> 16;
    }
  }

  operator float() const {
    float_bits_bf16 out;

    out.u = ((uint32_t)bits_) << 16;

    return out.f;
  }
};

#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                         \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_BFloat16 lhs, itype rhs) {            \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);     \
  }                                                                    \
  inline otype __operator__(itype lhs, _MLX_BFloat16 rhs) {            \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);     \
  }

#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, double, double, double);             \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, bool, float);         \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

#undef bfloat_binop

#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, double, double);      \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop

inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs) {
  return -static_cast<float>(lhs);
}

#define bfloat_inplace_op(__op__, __operator__)                              \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                    \
    return lhs;                                                              \
  }                                                                          \
  inline float& __operator__(float& lhs, _MLX_BFloat16 rhs) {                \
    lhs = lhs __op__ rhs;                                                    \
    return lhs;                                                              \
  }

bfloat_inplace_op(+, operator+=);
bfloat_inplace_op(-, operator-=);
bfloat_inplace_op(*, operator*=);
bfloat_inplace_op(/, operator/=);

#undef bfloat_inplace_op

#define bfloat_bitop(__op__, __operator__)                                  \
  inline _MLX_BFloat16 __operator__(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                                 \
    return out;                                                             \
  }                                                                         \
  inline _MLX_BFloat16 __operator__(_MLX_BFloat16 lhs, uint16_t rhs) {      \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs.bits_ __op__ rhs;                                       \
    return out;                                                             \
  }                                                                         \
  inline _MLX_BFloat16 __operator__(uint16_t lhs, _MLX_BFloat16 rhs) {      \
    _MLX_BFloat16 out;                                                      \
    out.bits_ = lhs __op__ rhs.bits_;                                       \
    return out;                                                             \
  }

bfloat_bitop(|, operator|);
bfloat_bitop(&, operator&);
bfloat_bitop(^, operator^);

#undef bfloat_bitop

#define bfloat_inplace_bitop(__op__, __operator__)                            \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                                   \
    return lhs;                                                               \
  }                                                                           \
  inline _MLX_BFloat16& __operator__(_MLX_BFloat16& lhs, uint16_t rhs) {      \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                         \
    return lhs;                                                               \
  }

bfloat_inplace_bitop(|, operator|=);
bfloat_inplace_bitop(&, operator&=);
bfloat_inplace_bitop(^, operator^=);

#undef bfloat_inplace_bitop

}
