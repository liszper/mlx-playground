#include "generator.h"
#include "kv_cache.h"
#include <iostream>
#include <vector>
#include <memory>

Generator::Generator(TransformerModel& model, Tokenizer& tokenizer, int max_length) 
    : model(model), tokenizer(tokenizer), max_length(max_length) {}

std::string Generator::generate(const std::string& prompt) {
    try {
        // Print prompt without newline and flush
        std::cout << prompt;
        std::cout.flush();

        // Tokenize prompt
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        array input = array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, int32);
        
        // Initialize KV cache
        const ModelArgs& args = model.get_args();
        int head_dim = args.hidden_size / args.num_attention_heads;
        std::vector<std::unique_ptr<KVCache>> cache;
        for (int i = 0; i < args.num_hidden_layers; i++) {
            cache.push_back(std::make_unique<KVCache>(args.num_key_value_heads, head_dim));
        }

        // Process all tokens except last one
        if (input.shape()[1] > 1) {
            std::vector<KVCache*> cache_ptrs;
            for (auto& c : cache) cache_ptrs.push_back(c.get());
            
            array context = slice(input, {0, 0}, {1, input.shape()[1] - 1});
            model.forward(context, &cache_ptrs);
            
            input = slice(input, {0, input.shape()[1] - 1}, {1, input.shape()[1]});
        }

        std::string generated_text;
        array y = input;

        // Generate tokens
        for (int i = 0; i < max_length; i++) {
            std::vector<KVCache*> cache_ptrs;
            for (auto& c : cache) cache_ptrs.push_back(c.get());
            
            array logits = model.forward(y, &cache_ptrs);
            array last_logits = slice(logits, {0, logits.shape()[1] - 1, 0}, 
                                           {1, logits.shape()[1], logits.shape()[2]});
            last_logits = reshape(last_logits, {1, -1});
            
            array next_token = argmax(last_logits, -1);
            eval(next_token);
            
            int token = next_token.item<int>();
            if (token == tokenizer.eos_token_id) {
                break;
            }
            
            // Decode and print the new token immediately
            std::string new_text = tokenizer.decode({token});
            std::cout << new_text;
            std::cout.flush();
            
            generated_text += new_text;
            y = array(&token, {1, 1}, int32);
        }
        std::cout << std::endl;
        
        return generated_text;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in generation: " << e.what() << std::endl;
        throw;
    }
} 

array _generate_step(Model& model, const array& y, std::vector<KVCache*>& cache) {
    try {
        // Get logits for next token prediction
        array logits = model.forward(y, &cache);
        
        // Take the last token's logits
        std::vector<int> start_indices = {0, logits.shape()[1] - 1, 0};  // [batch, last_token, vocab_start]
        std::vector<int> end_indices = {1, logits.shape()[1], logits.shape()[2]};  // [batch+1, last_token+1, vocab_end]
        array last_logits = slice(logits, start_indices, end_indices);
        
        // Reshape to 2D for argmax
        last_logits = reshape(last_logits, {1, -1});
        
        // Get next token
        array next_token = argmax(last_logits, -1);
        eval(next_token);
        
        return next_token;
    } catch (const std::exception& e) {
        std::cerr << "Error in _generate_step: " << e.what() << std::endl;
        throw;
    }
} 