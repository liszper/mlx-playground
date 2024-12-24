#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "kv_cache.h"
#include "rms_norm.h"
#include "linear.h"
#include "rope.h"
#include "embedding.h"
#include "model_args.h"
#include "feed_forward.h"
#include "model.h"
#include "optimizer.h"
#include "cross_entropy.h"

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
}; 