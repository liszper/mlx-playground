#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "kv_cache.cpp"
#include "rms_norm.cpp"
#include "linear.cpp"
#include "rope.cpp"
#include "embedding.cpp"
#include "model_args.cpp"
#include "feed_forward.cpp"
#include "model.cpp"
#include "optimizer.cpp"
#include "cross_entropy.cpp"

using namespace mlx::core;

class TransformerBlock {
private:
    int num_attention_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    float scale;
    
    std::unique_ptr<Linear> q_proj;
    std::unique_ptr<Linear> k_proj;
    std::unique_ptr<Linear> v_proj;
    std::unique_ptr<Linear> o_proj;
    std::unique_ptr<RMSNorm> input_layernorm;
    std::unique_ptr<RMSNorm> post_attention_layernorm;
    std::unique_ptr<FeedForward> mlp;
    std::unique_ptr<RoPE> rope;

public:
    TransformerBlock(const ModelArgs& args);
    array forward(const array& x, KVCache* cache = nullptr);
    std::unordered_map<std::string, array> parameters();
    void set_parameters(const std::unordered_map<std::string, array>& params);
};

class TransformerModel : public Model {
private:
    ModelArgs args;
    std::unique_ptr<Embedding> embed_tokens;
    std::vector<std::unique_ptr<TransformerBlock>> layers;
    std::unique_ptr<RMSNorm> norm;
    std::unique_ptr<Optimizer> optimizer;
    std::unordered_map<std::string, array> _parameters;

public:
    TransformerModel();
    array forward(const array& inputs, std::vector<KVCache*>* cache = nullptr) override;
    std::unordered_map<std::string, array>& parameters() override;
    void set_parameters(const std::unordered_map<std::string, array>& params) override;
    const ModelArgs& get_args() const;
    const std::vector<std::unique_ptr<TransformerBlock>>& get_layers() const;
    std::pair<array, std::unordered_map<std::string, array>> value_and_grad(
        const array& inputs, 
        const array& targets, 
        const array& lengths
    ) override;
    std::vector<array> state_arrays();
};

// Implementation

TransformerBlock::TransformerBlock(const ModelArgs& args) {
    // Calculate dimensions and scale in member initializer list
    num_attention_heads = args.num_attention_heads;
    num_kv_heads = args.num_key_value_heads;
    hidden_size = args.hidden_size;
    head_dim = hidden_size / num_attention_heads;
    scale = 1.0f / std::sqrt(head_dim);

    // Initialize all components in a single batch
    q_proj = std::make_unique<Linear>(hidden_size, num_attention_heads * head_dim);
    k_proj = std::make_unique<Linear>(hidden_size, num_kv_heads * head_dim);
    v_proj = std::make_unique<Linear>(hidden_size, num_kv_heads * head_dim);
    o_proj = std::make_unique<Linear>(num_attention_heads * head_dim, hidden_size);
    
    input_layernorm = std::make_unique<RMSNorm>(hidden_size);
    post_attention_layernorm = std::make_unique<RMSNorm>(hidden_size);
    mlp = std::make_unique<FeedForward>(args);
    rope = std::make_unique<RoPE>(head_dim);
}

array TransformerBlock::forward(const array& x, KVCache* cache) {
    // Apply input layer norm
    array h = input_layernorm->forward(x);
    
    // Get dimensions once
    const auto& shape = h.shape();
    int B = shape[0];
    int L = shape[1];
    
    // 1. Batch all projections together for better parallelization
    std::vector<array> projections = {
        q_proj->forward(h),
        k_proj->forward(h),
        v_proj->forward(h)
    };
    eval(projections);  // Force immediate evaluation
    
    // 2. Optimize reshape and transpose operations
    array queries = reshape(projections[0], {B, L, num_attention_heads, head_dim});
    array keys = reshape(projections[1], {B, L, num_kv_heads, head_dim});
    array values = reshape(projections[2], {B, L, num_kv_heads, head_dim});
    
    // 3. Batch transpose operations
    std::vector<array> transposed = {
        transpose(queries, {0, 2, 1, 3}),
        transpose(keys, {0, 2, 1, 3}),
        transpose(values, {0, 2, 1, 3})
    };
    eval(transposed);  // Force immediate evaluation
    queries = transposed[0];
    keys = transposed[1];
    values = transposed[2];
    
    // 4. Optimize RoPE application and cache handling
    if (cache) {
        queries = rope->forward(queries, cache->offset);
        keys = rope->forward(keys, cache->offset);
        std::tie(keys, values) = cache->update_and_fetch(keys, values);
    } else {
        queries = rope->forward(queries);
        keys = rope->forward(keys);
    }
    eval({queries, keys, values});  // Force immediate evaluation
    
    // 5. Optimize KV head repeat using concatenate
    if (num_kv_heads < num_attention_heads) {
        const int repeat_factor = num_attention_heads / num_kv_heads;
        keys = repeat(keys, 1, repeat_factor);
        values = repeat(values, 1, repeat_factor);
        eval({keys, values});
    }
    
    // 6. Compute attention and reshape in one step
    array attn_output = mlx::core::fast::scaled_dot_product_attention(
        queries, keys, values, scale, std::nullopt
    );
    eval({attn_output});
    
    // 7. Optimize final transformations
    attn_output = reshape(transpose(attn_output, {0, 2, 1, 3}), 
                         {B, L, num_attention_heads * head_dim});
    array output = o_proj->forward(attn_output) + x;
    eval({output});
    
    // 8. Final layer norm and MLP with forced evaluation
    array final_output = output + mlp->forward(post_attention_layernorm->forward(output));
    eval({final_output});
    
    return final_output;
}

std::unordered_map<std::string, array> TransformerBlock::parameters() {
    std::unordered_map<std::string, array> params;
    params.reserve(8);  // Reserve space for typical number of parameters
    
    // Use emplace instead of insert for better performance
    params.emplace("q_proj.weight", q_proj->get_weight());
    params.emplace("k_proj.weight", k_proj->get_weight());
    params.emplace("v_proj.weight", v_proj->get_weight());
    params.emplace("o_proj.weight", o_proj->get_weight());
    params.emplace("input_layernorm.weight", input_layernorm->get_weight());
    params.emplace("post_attention_layernorm.weight", post_attention_layernorm->get_weight());
    
    // Merge MLP parameters efficiently
    auto mlp_params = mlp->parameters();
    for (auto&& [name, param] : mlp_params) {
        params.emplace(std::move(name), std::move(param));
    }
    
    return params;
}

void TransformerBlock::set_parameters(const std::unordered_map<std::string, array>& params) {
    // Use structured bindings and find() for cleaner, efficient lookups
    if (auto it = params.find("q_proj.weight"); it != params.end()) {
        q_proj->set_weight(it->second);
    }
    
    if (auto it = params.find("k_proj.weight"); it != params.end()) {
        k_proj->set_weight(it->second);
    }
    
    if (auto it = params.find("v_proj.weight"); it != params.end()) {
        v_proj->set_weight(it->second);
    }
    
    if (auto it = params.find("o_proj.weight"); it != params.end()) {
        o_proj->set_weight(it->second);
    }
    
    if (auto it = params.find("input_layernorm.weight"); it != params.end()) {
        input_layernorm->set_weight(it->second);
    }
    
    if (auto it = params.find("post_attention_layernorm.weight"); it != params.end()) {
        post_attention_layernorm->set_weight(it->second);
    }
    
    // Update MLP parameters in batch
    mlp->set_parameters(params);
}

// TransformerModel implementations... 

TransformerModel::TransformerModel() {
    try {
        args = ModelArgs();  // Initialize with default values
        optimizer = std::unique_ptr<Optimizer>(new AdamW(1e-5));
        
        // Initialize embedding layer
        embed_tokens = std::make_unique<Embedding>(args.vocab_size, args.hidden_size);

        // Initialize transformer layers
        for (int i = 0; i < args.num_hidden_layers; i++) {
            layers.push_back(std::make_unique<TransformerBlock>(args));
        }
        
        // Initialize final normalization layer
        norm = std::make_unique<RMSNorm>(args.hidden_size);

        // Ensure all initializations are evaluated
        std::vector<array> params_to_eval;
        for (const auto& [_, param] : parameters()) {
            params_to_eval.push_back(param);
        }
        eval(params_to_eval);
    } catch (const std::exception& e) {
        std::cerr << "Error in Model initialization: " << e.what() << std::endl;
        throw;
    }
}

array TransformerModel::forward(const array& inputs, std::vector<KVCache*>* cache) {
    try {
        // 1. Pre-allocate vectors for batched operations
        const auto& shape = inputs.shape();
        const int batch_size = shape[0];
        const int seq_len = shape[1];
        std::vector<array> layer_outputs;
        layer_outputs.reserve(layers.size() + 2);  // +2 for embedding and final norm
        
        // 2. Optimize embedding with immediate evaluation
        array h = embed_tokens->forward(inputs);
        eval({h});
        layer_outputs.push_back(h);
        
        // 3. Process through transformer layers with batched evaluation
        for (size_t i = 0; i < layers.size(); i++) {
            h = layers[i]->forward(h, cache ? (*cache)[i] : nullptr);
            layer_outputs.push_back(h);
            
            // Evaluate every N layers (e.g., 4) to balance memory and parallelism
            if (i % 4 == 3 || i == layers.size() - 1) {
                eval(layer_outputs);
                layer_outputs.clear();
            }
        }
        
        // 4. Final normalization and projection with optimized memory usage
        h = norm->forward(h);
        array output = embed_tokens->as_linear(h);
        eval({output});
        
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in forward pass: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, array>& TransformerModel::parameters() {
    // Clear existing parameters with known capacity
    _parameters.clear();
    _parameters.reserve(layers.size() * 8 + 2);  // Approximate size based on typical layer params
    
    // Add embedding parameters with direct insertion
    _parameters.emplace("embed_tokens.weight", embed_tokens->get_weight());
    
    // Add layer parameters efficiently
    for (size_t i = 0; i < layers.size(); i++) {
        const auto& layer_params = layers[i]->parameters();
        const std::string layer_prefix = "layers." + std::to_string(i) + ".";
        
        // Pre-compute prefix to avoid repeated string concatenations
        for (const auto& [name, param] : layer_params) {
            _parameters.emplace(layer_prefix + name, param);
        }
    }
    
    // Add norm parameters
    _parameters.emplace("norm.weight", norm->get_weight());
    
    return _parameters;
}

void TransformerModel::set_parameters(const std::unordered_map<std::string, array>& params) {
    // Update embedding layer
    auto embed_it = params.find("embed_tokens.weight");
    if (embed_it != params.end()) {
        embed_tokens->set_weight(embed_it->second);
    }
    
    // Update transformer layers efficiently
    for (size_t i = 0; i < layers.size(); i++) {
        const std::string layer_prefix = "layers." + std::to_string(i) + ".";
        const size_t prefix_len = layer_prefix.length();
        
        // Create layer params map only if needed
        std::unordered_map<std::string, array> layer_params;
        
        // Single pass through params for this layer
        for (const auto& [name, param] : params) {
            if (name.compare(0, prefix_len, layer_prefix) == 0) {
                layer_params.emplace(name.substr(prefix_len), param);
            }
        }
        
        if (!layer_params.empty()) {
            layers[i]->set_parameters(layer_params);
        }
    }
    
    // Update final norm layer
    auto norm_it = params.find("norm.weight");
    if (norm_it != params.end()) {
        norm->set_weight(norm_it->second);
    }
    
    // Store all parameters
    _parameters = params;
}

const ModelArgs& TransformerModel::get_args() const { 
    return args; 
}

const std::vector<std::unique_ptr<TransformerBlock>>& TransformerModel::get_layers() const { 
    return layers; 
}

std::pair<array, std::unordered_map<std::string, array>> TransformerModel::value_and_grad(
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

std::vector<array> TransformerModel::state_arrays() {
    // Just return the parameters we're actually using
    std::vector<array> arrays;
    arrays.reserve(_parameters.size());  // Pre-allocate space
    
    for (const auto& [_, param] : _parameters) {
        arrays.push_back(param);
    }
    
    return arrays;
} 