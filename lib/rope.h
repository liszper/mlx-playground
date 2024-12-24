#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cmath>
#include <optional>

using namespace mlx::core;

class RoPE {
private:
    int dims;
    float scale;
    array _freqs;

public:
    RoPE(int dims, float scale = 1.0);
    array forward(const array& x, int offset = 0);
}; 