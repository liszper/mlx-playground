#pragma once

#include <memory>
#include <unordered_map>
#include "linear.h"
#include "model_args.h"
#include <mlx/mlx.h>

using namespace mlx::core;

// Helper function
inline array silu(const array& x) {
    return x * sigmoid(x);
}

class FeedForward {
private:
    std::unique_ptr<Linear> up_proj;
    std::unique_ptr<Linear> gate_proj;
    std::unique_ptr<Linear> down_proj;

public:
    FeedForward(const ModelArgs& args);
    array forward(const array& x);
    std::unordered_map<std::string, array> parameters();
    void set_parameters(const std::unordered_map<std::string, array>& params);
}; 