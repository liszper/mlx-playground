#include "model.h"
#include "cross_entropy.h"

LossResult cross_entropy_loss(Model& model, const array& inputs, const array& targets, const array& lengths) {
    array logits = model.forward(inputs);
    logits = astype(logits, float32);  // Match Python dtype

    array length_mask = expand_dims(arange(inputs.shape()[1]), 0) < expand_dims(lengths, 1);
    
    array ce = cross_entropy(logits, targets) * length_mask;
    int ntoks = sum(length_mask).item<int>();
    array loss = sum(ce) / static_cast<float>(ntoks);
    
    return {loss, ntoks};
}

std::pair<array, std::unordered_map<std::string, array>> Model::value_and_grad(
    const array& inputs, 
    const array& targets, 
    const array& lengths
) {
    // Create a wrapper function that captures model state and converts between parameter formats
    auto model_ptr = this;
    
    // Convert parameters map to vector
    auto params = parameters();
    std::vector<array> param_vec;
    std::vector<std::string> param_names;
    for (const auto& [name, param] : params) {
        param_names.push_back(name);
        param_vec.push_back(param);
    }
    
    // Create loss function that works with vectors and returns vector
    std::function<std::vector<array>(const std::vector<array>&)> loss_fn = 
        [model_ptr, &inputs, &targets, &lengths, &param_names](const std::vector<array>& params) -> std::vector<array> {
            // Reconstruct parameter map
            std::unordered_map<std::string, array> param_map;
            for (size_t i = 0; i < params.size(); i++) {
                param_map.emplace(param_names[i], params[i]);
            }
            model_ptr->set_parameters(param_map);
            return {cross_entropy_loss(*model_ptr, inputs, targets, lengths).loss};
        };
    
    // Call MLX's value_and_grad with vector indices
    std::vector<int> param_indices(param_vec.size());
    for (size_t i = 0; i < param_indices.size(); i++) {
        param_indices[i] = i;
    }
    
    auto value_and_grad_fn = mlx::core::value_and_grad(loss_fn, param_indices);
    auto result = value_and_grad_fn(param_vec);
    auto loss_vec = result.first;
    auto grad_vec = result.second;
    
    // Convert gradient vector back to map
    std::unordered_map<std::string, array> grads;
    for (size_t i = 0; i < grad_vec.size(); i++) {
        grads.emplace(param_names[i], grad_vec[i]);
    }
    
    return {loss_vec[0], grads};
} 