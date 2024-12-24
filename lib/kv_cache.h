#pragma once

#include <mlx/mlx.h>

using namespace mlx::core;

class KVCache {
public:
    array keys;
    array values;
    int offset;
    int step;
    int n_kv_heads;
    int head_dim;

    KVCache(int num_heads, int head_dim, int max_seq_len = 256);
    std::pair<array, array> update_and_fetch(const array& keys_in, const array& values_in);
}; 