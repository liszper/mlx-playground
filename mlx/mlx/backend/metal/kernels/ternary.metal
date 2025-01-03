#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/ternary_ops.h"
#include "mlx/backend/metal/kernels/ternary.h"

#define instantiate_ternary_all(op, tname, type)                  \
  instantiate_kernel("v_" #op #tname, ternary_v, type, op)        \
  instantiate_kernel("v2_" #op #tname, ternary_v2, type, op)      \
  instantiate_kernel("g_" #op #tname, ternary_g, type, op)        \
  instantiate_kernel("gn4_" #op #tname, ternary_g, type, op, 4)   \
  instantiate_kernel("g1_" #op #tname, ternary_g_nd1, type, op)   \
  instantiate_kernel("g2_" #op #tname, ternary_g_nd2, type, op)   \
  instantiate_kernel("g3_" #op #tname, ternary_g_nd3, type, op)

#define instantiate_ternary_types(op)               \
  instantiate_ternary_all(op, bool_, bool)          \
  instantiate_ternary_all(op, uint32, uint32_t)     \
  instantiate_ternary_all(op, uint64, uint64_t)     \
  instantiate_ternary_all(op, int32, int32_t)       \
  instantiate_ternary_all(op, int64, int64_t)       \
  instantiate_ternary_all(op, float32, float)       \
  instantiate_ternary_all(op, bfloat16, bfloat16_t)

instantiate_ternary_types(Select)
