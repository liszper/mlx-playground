#pragma once

#include <mlx/mlx.h>

using namespace mlx::core;

class RMSNorm {
private:
    array weight;
    float eps;

public:
    RMSNorm(int dim, float eps = 1e-6);
    array forward(const array& x);
    const array& get_weight() const;
    void set_weight(const array& w);
}; 