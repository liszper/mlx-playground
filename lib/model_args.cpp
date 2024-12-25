#pragma once

struct ModelArgs {
    int hidden_size = 3072;
    int num_hidden_layers = 3;
    int intermediate_size = 8192;
    int num_attention_heads = 24;
    int num_key_value_heads = 8;
    int vocab_size = 128256;
    float dropout = 0.1f;
};