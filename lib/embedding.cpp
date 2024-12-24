#pragma once

#include <mlx/mlx.h>
#include <cmath>

using namespace mlx::core;

class Embedding {
private:
    array weight;

public:
    Embedding(int num_embeddings, int embedding_dim)
        : weight(random::uniform(-std::sqrt(1.0f / embedding_dim), 
                               std::sqrt(1.0f / embedding_dim), 
                               {num_embeddings, embedding_dim}, 
                               float32)) {}

    array forward(const array& x) {
        return take(weight, x, 0);
    }

    array as_linear(const array& x) {
        auto input_shape = x.shape();
        array x_2d = reshape(x, {input_shape[0] * input_shape[1], input_shape[2]});
        array output = matmul(x_2d, transpose(weight));
        return reshape(output, {input_shape[0], input_shape[1], weight.shape()[0]});
    }

    const array& get_weight() const { 
        return weight; 
    }

    void set_weight(const array& w) { 
        weight = w; 
    }
}; 