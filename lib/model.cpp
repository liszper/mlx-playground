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
        // Create a wrapper function that captures model state and converts between parameter formats
        auto model_ptr = this;
        
        // Convert parameters map to vector more efficiently
        auto& params = parameters();  // Use reference to avoid copy
        std::vector<array> param_vec;
        std::vector<std::string> param_names;
        param_vec.reserve(params.size());  // Pre-allocate
        param_names.reserve(params.size());
        
        for (const auto& [name, param] : params) {
            param_names.push_back(name);
            param_vec.push_back(param);
        }
        
        // Create loss function that works with vectors and returns vector
        std::function<std::vector<array>(const std::vector<array>&)> loss_fn = 
            [model_ptr, &inputs, &targets, &lengths, &param_names](const std::vector<array>& params) -> std::vector<array> {
                // Reconstruct parameter map more efficiently
                std::unordered_map<std::string, array> param_map;
                param_map.reserve(params.size());  // Pre-allocate
                for (size_t i = 0; i < params.size(); i++) {
                    param_map.emplace(param_names[i], params[i]);
                }
                model_ptr->set_parameters(param_map);
                
                // Return loss directly without creating temporary vectors
                return {cross_entropy_loss(*model_ptr, inputs, targets, lengths).loss};
            };
        
        // Optimize param_indices creation
        std::vector<int> param_indices(param_vec.size());
        std::iota(param_indices.begin(), param_indices.end(), 0);  // More efficient than loop
        
        auto value_and_grad_fn = mlx::core::value_and_grad(loss_fn, param_indices);
        auto [loss_vec, grad_vec] = value_and_grad_fn(param_vec);  // Use structured binding
        
        // Convert gradient vector back to map more efficiently
        std::unordered_map<std::string, array> grads;
        grads.reserve(grad_vec.size());  // Pre-allocate
        for (size_t i = 0; i < grad_vec.size(); i++) {
            grads.emplace(param_names[i], std::move(grad_vec[i]));  // Use move semantics
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