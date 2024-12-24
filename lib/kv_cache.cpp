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
                    array trimmed_keys = slice(keys, {0, 0, 0, 0}, 
                                            {keys.shape()[0], keys.shape()[1], prev, keys.shape()[3]});
                    array trimmed_values = slice(values, {0, 0, 0, 0}, 
                                               {values.shape()[0], values.shape()[1], prev, values.shape()[3]});
                    keys = trimmed_keys;
                    values = trimmed_values;
                }
                keys = concatenate({keys, new_k}, 2);
                values = concatenate({values, new_v}, 2);
            } else {
                keys = new_k;
                values = new_v;
            }
        }

        // Update cache with new keys and values
        std::vector<array> key_parts;
        std::vector<array> value_parts;

        // Add existing cache content
        if (prev > 0) {
            key_parts.push_back(slice(keys, {0, 0, 0, 0}, 
                                    {keys.shape()[0], keys.shape()[1], prev, keys.shape()[3]}));
            value_parts.push_back(slice(values, {0, 0, 0, 0}, 
                                      {values.shape()[0], values.shape()[1], prev, values.shape()[3]}));
        }

        // Add new inputs
        key_parts.push_back(keys_in);
        value_parts.push_back(values_in);

        // Add remaining cache if any
        if (keys.shape()[2] > prev + keys_in.shape()[2]) {
            key_parts.push_back(slice(keys, 
                                    {0, 0, prev + keys_in.shape()[2], 0}, 
                                    {keys.shape()[0], keys.shape()[1], keys.shape()[2], keys.shape()[3]}));
            value_parts.push_back(slice(values, 
                                      {0, 0, prev + values_in.shape()[2], 0}, 
                                      {values.shape()[0], values.shape()[1], values.shape()[2], values.shape()[3]}));
        }

        // Combine all parts
        keys = concatenate(key_parts, 2);
        values = concatenate(value_parts, 2);
        
        offset += keys_in.shape()[2];
        
        // Return the complete cache up to current offset
        return {
            slice(keys, {0, 0, 0, 0}, 
                  {keys.shape()[0], keys.shape()[1], offset, keys.shape()[3]}),
            slice(values, {0, 0, 0, 0}, 
                  {values.shape()[0], values.shape()[1], offset, values.shape()[3]})
        };
    }
}; 