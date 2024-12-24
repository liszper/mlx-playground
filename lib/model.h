#pragma once

#include <mlx/mlx.h>
#include <unordered_map>
#include <vector>
#include "kv_cache.h"

using namespace mlx::core;

struct LossResult {
    array loss;
    int num_tokens;
};

// Forward declarations
class Model;

LossResult cross_entropy_loss(Model& model, const array& inputs, const array& targets, const array& lengths);

class Model {
public:
    virtual ~Model() = default;
    virtual array forward(const array& inputs, std::vector<KVCache*>* cache = nullptr) = 0;
    virtual std::unordered_map<std::string, array>& parameters() = 0;
    virtual void set_parameters(const std::unordered_map<std::string, array>& params) = 0;
    virtual std::pair<array, std::unordered_map<std::string, array>> value_and_grad(
        const array& inputs, 
        const array& targets, 
        const array& lengths
    );
};