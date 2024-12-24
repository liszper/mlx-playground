#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

enum class CopyType {
  Scalar,

  Vector,

  General,

  GeneralGeneral
};

void copy(const array& src, array& dst, CopyType ctype);
void copy_inplace(const array& src, array& dst, CopyType ctype);

template <typename stride_t>
void copy_inplace(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype);

}
