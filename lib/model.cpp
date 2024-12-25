#pragma once

#include <mlx/mlx.h>
#include <unordered_map>
#include <vector>
#include <numeric>
#include "kv_cache.cpp"
#include "cross_entropy.cpp"

using namespace mlx::core;

struct LossResult {
    array loss;
    int num_tokens;
};

// Forward declare cross_entropy_loss
inline LossResult cross_entropy_loss(class Model& model, const array& inputs, const array& targets, const array& lengths);

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
    ) {
        // Optimize parameter conversion
        auto& params = parameters();
        std::vector<array> param_vec;
        std::vector<std::string> param_names;
        param_vec.reserve(params.size());
        param_names.reserve(params.size());
        
        // Single pass through parameters
        for (const auto& [name, param] : params) {
            param_names.push_back(name);
            param_vec.push_back(param);
        }
        
        // Efficient loss function
        auto loss_fn = [this, &inputs, &targets, &lengths, &param_names](
            const std::vector<array>& params
        ) -> std::vector<array> {
            // Reconstruct parameters efficiently
            std::unordered_map<std::string, array> param_map;
            param_map.reserve(params.size());
            
            for (size_t i = 0; i < params.size(); i++) {
                param_map.emplace(param_names[i], params[i]);
            }
            
            set_parameters(param_map);
            auto loss_result = cross_entropy_loss(*this, inputs, targets, lengths);
            return {loss_result.loss};
        };
        
        // Generate parameter indices efficiently
        std::vector<int> param_indices(param_vec.size());
        std::iota(param_indices.begin(), param_indices.end(), 0);
        
        // Compute gradients
        auto value_and_grad_fn = mlx::core::value_and_grad(loss_fn, param_indices);
        auto [loss_vec, grad_vec] = value_and_grad_fn(param_vec);
        
        // Convert gradients to map efficiently
        std::unordered_map<std::string, array> grads;
        grads.reserve(grad_vec.size());
        
        for (size_t i = 0; i < grad_vec.size(); i++) {
            grads.emplace(param_names[i], grad_vec[i]);
        }
        
        return {loss_vec[0], grads};
    }
};

// Implement cross_entropy_loss after Model definition
inline LossResult cross_entropy_loss(Model& model, const array& inputs, const array& targets, const array& lengths) {
    array logits = model.forward(inputs);
    logits = astype(logits, float32);  // Match Python dtype

    array length_mask = expand_dims(arange(inputs.shape()[1]), 0) < expand_dims(lengths, 1);
    
    array ce = cross_entropy(logits, targets) * length_mask;
    int ntoks = sum(length_mask).item<int>();
    array loss = sum(ce) / static_cast<float>(ntoks);
    
    return {loss, ntoks};
}