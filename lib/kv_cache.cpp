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

    KVCache(int num_heads, int head_dim, int max_seq_len = 256)
        : keys(zeros({0}, float32)),
          values(zeros({0}, float32)),
          offset(0),
          step(max_seq_len),
          n_kv_heads(num_heads),
          head_dim(head_dim) {}

    std::pair<array, array> update_and_fetch(const array& keys_in, const array& values_in) {
        int prev = offset;
        
        // Initialize or resize cache if needed
        if (keys.size() == 0 || (prev + keys_in.shape()[2]) > keys.shape()[2]) {
            int B = keys_in.shape()[0];
            int n_steps = (step + keys_in.shape()[2] - 1) / step;  // Ceiling division
            
            array new_k = zeros({B, n_kv_heads, n_steps * step, head_dim}, keys_in.dtype());
            array new_v = zeros({B, n_kv_heads, n_steps * step, head_dim}, values_in.dtype());
            
            if (keys.size() > 0) {
                if (prev % step != 0) {
                    std::vector<int> start = {0, 0, 0, 0};
                    std::vector<int> stop = {keys.shape()[0], keys.shape()[1], prev, keys.shape()[3]};
                    keys = slice(keys, start, stop);
                    values = slice(values, start, stop);
                }
                // Retain arrays in memory by using inplace operations
                keys = concatenate({keys, new_k}, 2);
                values = concatenate({values, new_v}, 2);
                eval({keys, values});  // Ensure arrays are evaluated and retained
            } else {
                keys = new_k;
                values = new_v;
                eval({keys, values});
            }
        }

        offset += keys_in.shape()[2];
        
        // Use inplace updates and evaluate immediately
        keys = slice_update(keys, keys_in, {0, 0, prev, 0}, 
                          {keys.shape()[0], keys.shape()[1], offset, keys.shape()[3]});
        values = slice_update(values, values_in, {0, 0, prev, 0},
                            {values.shape()[0], values.shape()[1], offset, values.shape()[3]});
        eval({keys, values});

        // Return slices of the cached arrays
        return {
            slice(keys, {0, 0, 0, 0}, {keys.shape()[0], keys.shape()[1], offset, keys.shape()[3]}),
            slice(values, {0, 0, 0, 0}, {values.shape()[0], values.shape()[1], offset, values.shape()[3]})
        };
    }
}; 