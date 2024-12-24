#include "linear.h"
#include <cmath>

Linear::Linear(int input_dims, int output_dims) 
    : weight(random::uniform(-std::sqrt(1.0f / input_dims), 
                           std::sqrt(1.0f / input_dims), 
                           {output_dims, input_dims},  // Match Python dimensions
                           bfloat16)) {}  // Use bfloat16 to match Python

array Linear::forward(const array& x) {
    // Simplified matrix multiplication matching Python
    return matmul(x, transpose(weight));
}

const array& Linear::get_weight() const { 
    return weight; 
}

void Linear::set_weight(const array& w) { 
    weight = w; 
} 