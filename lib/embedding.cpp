#include "embedding.h"
#include <cmath>

Embedding::Embedding(int num_embeddings, int embedding_dim) 
    : weight(random::uniform(-std::sqrt(1.0f / embedding_dim), 
                           std::sqrt(1.0f / embedding_dim), 
                           {num_embeddings, embedding_dim}, 
                           float32)) {}

array Embedding::forward(const array& x) {
    return take(weight, x, 0);
}

array Embedding::as_linear(const array& x) {
    auto input_shape = x.shape();
    array x_2d = reshape(x, {input_shape[0] * input_shape[1], input_shape[2]});
    array output = matmul(x_2d, transpose(weight));
    return reshape(output, {input_shape[0], input_shape[1], weight.shape()[0]});
}

const array& Embedding::get_weight() const { 
    return weight; 
}

void Embedding::set_weight(const array& w) { 
    weight = w; 
} 