#pragma once

#include <mlx/mlx.h>

using namespace mlx::core;

class Embedding {
private:
    array weight;

public:
    Embedding(int num_embeddings, int embedding_dim);
    array forward(const array& x);
    array as_linear(const array& x);
    const array& get_weight() const;
    void set_weight(const array& w);
}; 