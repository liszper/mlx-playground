#pragma once

#include <memory>
#include <unordered_map>
#include "linear.cpp"
#include "model_args.cpp"
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
    inline FeedForward(const ModelArgs& args) {
        up_proj = std::make_unique<Linear>(args.hidden_size, args.intermediate_size);
        gate_proj = std::make_unique<Linear>(args.hidden_size, args.intermediate_size);
        down_proj = std::make_unique<Linear>(args.intermediate_size, args.hidden_size);
    }

    inline array forward(const array& x) {
        array up = up_proj->forward(x);
        array gate = gate_proj->forward(x);
        array activated = silu(gate) * up;
        return down_proj->forward(activated);
    }

    inline std::unordered_map<std::string, array> parameters() {
        return {
            {"up_proj.weight", up_proj->get_weight()},
            {"gate_proj.weight", gate_proj->get_weight()},
            {"down_proj.weight", down_proj->get_weight()}
        };
    }

    inline void set_parameters(const std::unordered_map<std::string, array>& params) {
        if (params.find("up_proj.weight") != params.end()) {
            up_proj->set_weight(params.at("up_proj.weight"));
        }
        if (params.find("gate_proj.weight") != params.end()) {
            gate_proj->set_weight(params.at("gate_proj.weight"));
        }
        if (params.find("down_proj.weight") != params.end()) {
            down_proj->set_weight(params.at("down_proj.weight"));
        }
    }
}; 