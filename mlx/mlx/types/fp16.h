#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define __MLX_HALF_NAN__ 0x7D00

namespace mlx::core {

namespace {
union float_bits_fp16 {
  float f;
  uint32_t u;
};
}

struct _MLX_Float16 {
  uint16_t bits_;

  _MLX_Float16() = default;

  _MLX_Float16(_MLX_Float16 const&) = default;

  _MLX_Float16& operator=(std::vector<bool>::reference x) {
    bits_ = x;
    return *this;
  }

  _MLX_Float16& operator=(const float& x) {
    return (*this = _MLX_Float16(x));
  }

  _MLX_Float16(const float& x) : bits_(0) {

    float_bits_fp16 in;

    in.f = x;

    uint32_t x_sign_32 = in.u & uint32_t(0x80000000);
    uint16_t x_sign_16 = (x_sign_32 >> 16);

    if (std::isnan(x)) {
      bits_ = x_sign_16 | uint16_t(__MLX_HALF_NAN__);
    } else {
      float_bits_fp16 inf_scale, zero_scale, magic_bits;

      uint32_t x_expo_32 = in.u & uint32_t(0x7f800000);
      uint32_t max_expo_32 = uint32_t(0x38800000);
      x_expo_32 = x_expo_32 < max_expo_32 ? max_expo_32 : x_expo_32;
      x_expo_32 += uint32_t(15) << 23;

      inf_scale.u = uint32_t(0x77800000);
      zero_scale.u = uint32_t(0x08800000);

      magic_bits.u = x_expo_32;
      magic_bits.f += (std::abs(x) * inf_scale.f) * zero_scale.f;

      uint32_t x_expo_16 = ((magic_bits.u >> 13) & uint32_t(0x7c00));

      uint32_t x_mant_16 = magic_bits.u & uint32_t(0x0fff);

      bits_ = (x_sign_16 | uint16_t(x_expo_16 + x_mant_16));
    }
  }

  operator float() const {

    float_bits_fp16 out;

    uint32_t x_sign_32 = (bits_ << 16) & uint32_t(0x80000000);
    uint32_t base = (bits_ << 16);
    uint32_t two_base = base + base;

    uint32_t denorm_max = 1u << 27;
    if (two_base < denorm_max) {
      out.u = uint32_t(126) << 23;
      out.u |= (two_base >> 17);
      out.f -= 0.5f;
    } else {
      out.u = uint32_t(0xE0) << 23;
      out.u += (two_base >> 4);
      float out_unscaled = out.f;
      out.u = uint32_t(0x7800000);
      out.f *= out_unscaled;
    }

    out.u |= x_sign_32;

    return out.f;
  }
};

#define half_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  inline otype __operator__(atype lhs, btype rhs) {                       \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

#define half_binop_helper(__op__, __operator__, otype, itype, ctype) \
  inline otype __operator__(_MLX_Float16 lhs, itype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);   \
  }                                                                  \
  inline otype __operator__(itype lhs, _MLX_Float16 rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);   \
  }

#define half_binop(__op__, __operator__)                                      \
  half_binop_base(                                                            \
      __op__, __operator__, _MLX_Float16, _MLX_Float16, _MLX_Float16, float); \
  half_binop_helper(__op__, __operator__, float, float, float);               \
  half_binop_helper(__op__, __operator__, double, double, double);            \
  half_binop_helper(__op__, __operator__, _MLX_Float16, bool, float);         \
  half_binop_helper(__op__, __operator__, _MLX_Float16, int32_t, float);      \
  half_binop_helper(__op__, __operator__, _MLX_Float16, uint32_t, float);     \
  half_binop_helper(__op__, __operator__, _MLX_Float16, int64_t, float);      \
  half_binop_helper(__op__, __operator__, _MLX_Float16, uint64_t, float);

half_binop(+, operator+);
half_binop(-, operator-);
half_binop(*, operator*);
half_binop(/, operator/);

#undef half_binop

#define half_compop(__op__, __operator__)                             \
  half_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_Float16, _MLX_Float16, float); \
  half_binop_helper(__op__, __operator__, bool, float, float);        \
  half_binop_helper(__op__, __operator__, bool, double, double);      \
  half_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  half_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  half_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  half_binop_helper(__op__, __operator__, bool, uint64_t, float);

half_compop(>, operator>);
half_compop(<, operator<);
half_compop(>=, operator>=);
half_compop(<=, operator<=);
half_compop(==, operator==);
half_compop(!=, operator!=);

#undef half_compop

inline _MLX_Float16 operator-(_MLX_Float16 lhs) {
  return -static_cast<float>(lhs);
}

#define half_inplace_op(__op__, __operator__)                              \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, const float& rhs) { \
    lhs = lhs __op__ rhs;                                                  \
    return lhs;                                                            \
  }                                                                        \
  inline float& __operator__(float& lhs, _MLX_Float16 rhs) {               \
    lhs = lhs __op__ rhs;                                                  \
    return lhs;                                                            \
  }

half_inplace_op(+, operator+=);
half_inplace_op(-, operator-=);
half_inplace_op(*, operator*=);
half_inplace_op(/, operator/=);

#undef half_inplace_op

#define half_bitop(__op__, __operator__)                                 \
  inline _MLX_Float16 __operator__(_MLX_Float16 lhs, _MLX_Float16 rhs) { \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs.bits_;                              \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float16 __operator__(_MLX_Float16 lhs, uint16_t rhs) {     \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs.bits_ __op__ rhs;                                    \
    return out;                                                          \
  }                                                                      \
  inline _MLX_Float16 __operator__(uint16_t lhs, _MLX_Float16 rhs) {     \
    _MLX_Float16 out;                                                    \
    out.bits_ = lhs __op__ rhs.bits_;                                    \
    return out;                                                          \
  }

half_bitop(|, operator|);
half_bitop(&, operator&);
half_bitop(^, operator^);

#undef half_bitop

#define half_inplace_bitop(__op__, __operator__)                           \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, _MLX_Float16 rhs) { \
    lhs.bits_ = lhs.bits_ __op__ rhs.bits_;                                \
    return lhs;                                                            \
  }                                                                        \
  inline _MLX_Float16& __operator__(_MLX_Float16& lhs, uint16_t rhs) {     \
    lhs.bits_ = lhs.bits_ __op__ rhs;                                      \
    return lhs;                                                            \
  }

half_inplace_bitop(|, operator|=);
half_inplace_bitop(&, operator&=);
half_inplace_bitop(^, operator^=);

#undef half_inplace_bitop

}
