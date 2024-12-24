#include "feed_forward.h"

FeedForward::FeedForward(const ModelArgs& args) {
    up_proj = std::make_unique<Linear>(args.hidden_size, args.intermediate_size);
    gate_proj = std::make_unique<Linear>(args.hidden_size, args.intermediate_size);
    down_proj = std::make_unique<Linear>(args.intermediate_size, args.hidden_size);
}

array FeedForward::forward(const array& x) {
    array up = up_proj->forward(x);
    array gate = gate_proj->forward(x);
    array activated = silu(gate) * up;
    return down_proj->forward(activated);
}

std::unordered_map<std::string, array> FeedForward::parameters() {
    return {
        {"up_proj.weight", up_proj->get_weight()},
        {"gate_proj.weight", gate_proj->get_weight()},
        {"down_proj.weight", down_proj->get_weight()}
    };
}

void FeedForward::set_parameters(const std::unordered_map<std::string, array>& params) {
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