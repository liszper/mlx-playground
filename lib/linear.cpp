#pragma once

#include <mlx/mlx.h>
#include <cmath>

using namespace mlx::core;

// Linear layer implementation
class Linear {
private:
    array weight;

public:
    Linear(int input_dims, int output_dims) 
        : weight(random::uniform(-std::sqrt(1.0f / input_dims), 
                               std::sqrt(1.0f / input_dims), 
                               {output_dims, input_dims},  // Match Python dimensions
                               bfloat16)) {}  // Use bfloat16 to match Python

    array forward(const array& x) {
        // Simplified matrix multiplication matching Python
        return matmul(x, transpose(weight));
    }

    const array& get_weight() const { 
        return weight; 
    }

    void set_weight(const array& w) { 
        weight = w; 
    }
}; 