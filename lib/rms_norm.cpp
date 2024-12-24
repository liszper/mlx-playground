#include "rms_norm.h"

RMSNorm::RMSNorm(int dim, float eps) 
    : weight(ones({dim}, float32)), 
      eps(eps) {}

array RMSNorm::forward(const array& x) {
    array variance = mean(square(x), -1, true);
    return x * rsqrt(variance + eps) * weight;
}

const array& RMSNorm::get_weight() const { 
    return weight; 
}

void RMSNorm::set_weight(const array& w) { 
    weight = w; 
} 