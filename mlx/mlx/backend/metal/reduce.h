#pragma once

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mlx::core {

using metal::CommandEncoder;

void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s,
    std::vector<array>& copies);

void row_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s);

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s);

}
