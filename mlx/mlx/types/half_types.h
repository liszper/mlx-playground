#pragma once
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

#include <arm_fp16.h>
namespace mlx::core {
typedef __fp16 float16_t;
}

#else

#define ADD_HALF_BINOPS
#include "mlx/types/fp16.h"
namespace mlx::core {
typedef struct _MLX_Float16 float16_t;
}

#endif
#ifdef __ARM_FEATURE_BF16

#include <arm_bf16.h>
namespace mlx::core {
typedef __bf16 bfloat16_t;
}

#else

#define ADD_HALF_BINOPS
#include "mlx/types/bf16.h"
namespace mlx::core {
typedef struct _MLX_BFloat16 bfloat16_t;
}

#endif

#ifdef ADD_HALF_BINOPS
namespace mlx::core {

#define fp16_bf16_binop_helper(__op__, __operator__)               \
  inline float __operator__(float16_t lhs, bfloat16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }                                                                \
  inline float __operator__(bfloat16_t lhs, float16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }

fp16_bf16_binop_helper(+, operator+)
fp16_bf16_binop_helper(-, operator-)
fp16_bf16_binop_helper(*, operator*)
fp16_bf16_binop_helper(/, operator/)

}
#endif
