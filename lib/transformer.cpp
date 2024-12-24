#include "transformer.h"
#include <cmath>
#include <iostream>

TransformerBlock::TransformerBlock(const ModelArgs& args) : 
    num_attention_heads(args.num_attention_heads),
    num_kv_heads(args.num_key_value_heads),
    head_dim(args.hidden_size / args.num_attention_heads),
    hidden_size(args.hidden_size),
    scale(std::sqrt(1.0f / head_dim)) {
    
    // Initialize projections
    q_proj = std::make_unique<Linear>(hidden_size, num_attention_heads * head_dim);
    k_proj = std::make_unique<Linear>(hidden_size, num_kv_heads * head_dim);
    v_proj = std::make_unique<Linear>(hidden_size, num_kv_heads * head_dim);
    o_proj = std::make_unique<Linear>(num_attention_heads * head_dim, hidden_size);
    
    // Initialize layer norms
    input_layernorm = std::make_unique<RMSNorm>(hidden_size);
    post_attention_layernorm = std::make_unique<RMSNorm>(hidden_size);
    
    // Initialize feed forward network
    mlp = std::make_unique<FeedForward>(args);
    
    // Initialize RoPE
    rope = std::make_unique<RoPE>(head_dim);
}

array TransformerBlock::forward(const array& x, KVCache* cache) {
    // Apply input layer norm
    array h = input_layernorm->forward(x);
    
    // Get batch size and sequence length
    int B = h.shape()[0];
    int L = h.shape()[1];
    
    // Project to Q, K, V
    array queries = q_proj->forward(h);
    array keys = k_proj->forward(h);
    array values = v_proj->forward(h);
    
    // Reshape to separate heads
    queries = reshape(queries, {B, L, num_attention_heads, head_dim});
    keys = reshape(keys, {B, L, num_kv_heads, head_dim});
    values = reshape(values, {B, L, num_kv_heads, head_dim});
    
    // Transpose for attention computation
    queries = transpose(queries, {0, 2, 1, 3}); // [B, num_heads, L, head_dim]
    keys = transpose(keys, {0, 2, 1, 3});       // [B, num_kv_heads, L, head_dim]
    values = transpose(values, {0, 2, 1, 3});   // [B, num_kv_heads, L, head_dim]
    
    // Apply RoPE
    if (cache) {
        queries = rope->forward(queries, cache->offset);
        keys = rope->forward(keys, cache->offset);
        // Update cache and get full keys/values
        auto [cached_keys, cached_values] = cache->update_and_fetch(keys, values);
        keys = cached_keys;
        values = cached_values;
    } else {
        queries = rope->forward(queries);
        keys = rope->forward(keys);
    }
    
    // Repeat keys and values if num_kv_heads < num_attention_heads
    if (num_kv_heads < num_attention_heads) {
        int repeat_factor = num_attention_heads / num_kv_heads;
        std::vector<array> repeated_keys(repeat_factor, keys);
        std::vector<array> repeated_values(repeat_factor, values);
        keys = concatenate(repeated_keys, 1);
        values = concatenate(repeated_values, 1);
    }
    
    // Compute scaled dot-product attention
    array attn_weights = matmul(queries, transpose(keys, {0, 1, 3, 2})) * scale;
    
    // Apply softmax
    attn_weights = softmax(attn_weights, -1);
    
    // Compute attention output
    array attn_output = matmul(attn_weights, values);
    
    // Reshape and transpose back
    attn_output = transpose(attn_output, {0, 2, 1, 3});
    attn_output = reshape(attn_output, {B, L, num_attention_heads * head_dim});
    
    // Project to output dimension
    array output = o_proj->forward(attn_output);
    
    // First residual connection
    output = output + x;
    
    // Apply post attention layer norm
    array h2 = post_attention_layernorm->forward(output);
    
    // Feed forward network
    array ff_output = mlp->forward(h2);
    
    // Second residual connection
    return output + ff_output;
}

std::unordered_map<std::string, array> TransformerBlock::parameters() {
    std::unordered_map<std::string, array> params;
    
    // Add parameters using emplace
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("q_proj.weight"),
                  std::forward_as_tuple(q_proj->get_weight()));
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("k_proj.weight"),
                  std::forward_as_tuple(k_proj->get_weight()));
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("v_proj.weight"),
                  std::forward_as_tuple(v_proj->get_weight()));
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("o_proj.weight"),
                  std::forward_as_tuple(o_proj->get_weight()));
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("input_layernorm.weight"),
                  std::forward_as_tuple(input_layernorm->get_weight()));
    params.emplace(std::piecewise_construct,
                  std::forward_as_tuple("post_attention_layernorm.weight"),
                  std::forward_as_tuple(post_attention_layernorm->get_weight()));
    
    // Add MLP parameters
    auto mlp_params = mlp->parameters();
    for (const auto& [name, param] : mlp_params) {
        params.emplace(std::piecewise_construct,
                     std::forward_as_tuple(name),
                     std::forward_as_tuple(param));
    }
    
    return params;
}

void TransformerBlock::set_parameters(const std::unordered_map<std::string, array>& params) {
    auto it = params.find("q_proj.weight");
    if (it != params.end()) {
        q_proj->set_weight(it->second);
    }
    
    it = params.find("k_proj.weight");
    if (it != params.end()) {
        k_proj->set_weight(it->second);
    }
    
    it = params.find("v_proj.weight");
    if (it != params.end()) {
        v_proj->set_weight(it->second);
    }
    
    it = params.find("o_proj.weight");
    if (it != params.end()) {
        o_proj->set_weight(it->second);
    }
    
    it = params.find("input_layernorm.weight");
    if (it != params.end()) {
        input_layernorm->set_weight(it->second);
    }
    
    it = params.find("post_attention_layernorm.weight");
    if (it != params.end()) {
        post_attention_layernorm->set_weight(it->second);
    }
    
    // Update MLP parameters
    mlp->set_parameters(params);
}

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
        // First embed the tokens
        array h = embed_tokens->forward(inputs);
        
        // Process through transformer layers
        for (size_t i = 0; i < layers.size(); i++) {
            h = layers[i]->forward(h, cache ? (*cache)[i] : nullptr);
        }
        
        // Final normalization
        h = norm->forward(h);
        
        // Project to vocabulary
        array output = embed_tokens->as_linear(h);
        
        return output;
    } catch (const std::exception& e) {
        std::cerr << "Error in forward pass: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, array>& TransformerModel::parameters() {
    // Clear existing parameters
    _parameters.clear();
    
    // Add embedding parameters
    _parameters.insert({"embed_tokens.weight", embed_tokens->get_weight()});
    
    // Add layer parameters
    for (size_t i = 0; i < layers.size(); i++) {
        auto layer_params = layers[i]->parameters();
        for (const auto& [name, param] : layer_params) {
            _parameters.insert({"layers." + std::to_string(i) + "." + name, param});
        }
    }
    
    // Add norm parameters
    _parameters.insert({"norm.weight", norm->get_weight()});
    
    return _parameters;
}

void TransformerModel::set_parameters(const std::unordered_map<std::string, array>& params) {
    // Store the parameters
    for (const auto& [name, param] : params) {
        _parameters.insert_or_assign(name, param);
    }
    
    // Update embedding layer
    if (params.find("embed_tokens.weight") != params.end()) {
        embed_tokens->set_weight(params.at("embed_tokens.weight"));
    }
    
    // Update transformer layers
    for (size_t i = 0; i < layers.size(); i++) {
        std::string layer_prefix = "layers." + std::to_string(i) + ".";
        std::unordered_map<std::string, array> layer_params;
        
        // Collect all parameters for this layer
        for (const auto& [name, param] : params) {
            if (name.find(layer_prefix) == 0) {
                // Remove the layer prefix to get the parameter name
                std::string param_name = name.substr(layer_prefix.length());
                layer_params.insert_or_assign(param_name, param);
            }
        }
        
        // Update layer parameters
        if (!layer_params.empty()) {
            layers[i]->set_parameters(layer_params);
        }
    }
    
    // Update final normalization layer
    if (params.find("norm.weight") != params.end()) {
        norm->set_weight(params.at("norm.weight"));
    }
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