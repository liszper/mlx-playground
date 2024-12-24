#pragma once

#include <optional>

#include "mlx/utils.h"

namespace mlx::core::fast {

array rms_norm(
    const array& x,
    const array& weight,
    float eps,
    StreamOrDevice s = {});

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask = std::nullopt,
    const std::optional<int> memory_efficient_threshold = std::nullopt,
    StreamOrDevice s = {});

typedef std::variant<int, bool, Dtype> TemplateArg;

}
