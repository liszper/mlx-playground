#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/arange.h"

#define instantiate_arange(tname, type)                                 \
  template [[host_name("arange" #tname)]] [[kernel]] void arange<type>( \
      constant const type& start,                                       \
      constant const type& step,                                        \
      device type* out,                                                 \
      uint index [[thread_position_in_grid]]);

instantiate_arange(uint32, uint32_t)
instantiate_arange(uint64, uint64_t)
instantiate_arange(int32, int32_t)
instantiate_arange(int64, int64_t)
instantiate_arange(float32, float)
instantiate_arange(bfloat16, bfloat16_t)
