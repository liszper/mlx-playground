#include "rope.h"

RoPE::RoPE(int dims, float scale) 
    : dims(dims), 
      scale(scale),
      _freqs(zeros({0}, float32)) {
    float base = 500000.0f;
    
    // Create freqs array directly matching Python
    std::vector<float> freqs(dims/2);
    for (int i = 0; i < dims/2; i++) {
        freqs[i] = std::pow(base, -static_cast<float>(2*i) / dims);
    }
    _freqs = array(freqs.data(), {dims/2}, float32);
    
    float factor = 32.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 4.0f;
    int old_context_len = 8192;
    
    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;
    
    array wavelens = 2.0f * M_PI * _freqs;
    
    // Match Python's boolean operations
    array high_freq_mask = wavelens > low_freq_wavelen;
    _freqs = where(high_freq_mask, _freqs * factor, _freqs);
    
    array is_medium_freq = logical_and(
        wavelens > high_freq_wavelen,
        wavelens < low_freq_wavelen
    );
    
    array smooth_factors = (old_context_len / wavelens - low_freq_factor) / 
                         (high_freq_factor - low_freq_factor);
    array smooth_freqs = _freqs / ((1.0f - smooth_factors) / factor + smooth_factors);
    
    _freqs = where(is_medium_freq, smooth_freqs, _freqs) * scale;
}

array RoPE::forward(const array& x, int offset) {
    return mlx::core::fast::rope(x, dims, false, std::nullopt, scale, offset, _freqs);
} 