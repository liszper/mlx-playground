#pragma once

#include <mlx/mlx.h>

using namespace mlx::core;

class RMSNorm {
private:
    array weight;
    float eps;

public:
    RMSNorm(int dim, float eps = 1e-6) 
        : weight(ones({dim}, float32)), 
          eps(eps) {}

    array forward(const array& x) {
        array variance = mean(square(x), -1, true);
        return x * rsqrt(variance + eps) * weight;
    }

    const array& get_weight() const { 
        return weight; 
    }

    void set_weight(const array& w) { 
        weight = w; 
    }
}; 