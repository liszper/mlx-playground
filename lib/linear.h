#pragma once

#include <mlx/mlx.h>

using namespace mlx::core;

// Linear layer implementation
class Linear {
private:
    array weight;

public:
    Linear(int input_dims, int output_dims);
    array forward(const array& x);
    const array& get_weight() const;
    void set_weight(const array& w);
}; 