#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include "model.cpp"
#include "tokenizer.cpp"
#include "model_args.cpp"
#include "transformer.cpp"
#include "kv_cache.cpp"

class Generator {
private:
    TransformerModel& model;
    Tokenizer& tokenizer;
    int max_length;
    
    array _generate_step(Model& model, const array& y, std::vector<KVCache*>& cache) {
        try {
            array logits = model.forward(y, &cache);
            
            std::vector<int> start_indices = {0, logits.shape()[1] - 1, 0};
            std::vector<int> end_indices = {1, logits.shape()[1], logits.shape()[2]};
            array last_logits = slice(logits, start_indices, end_indices);
            
            last_logits = reshape(last_logits, {1, -1});
            array next_token = argmax(last_logits, -1);
            eval(next_token);
            
            return next_token;
        } catch (const std::exception& e) {
            std::cerr << "Error in _generate_step: " << e.what() << std::endl;
            throw;
        }
    }
    
public:
    Generator(TransformerModel& model, Tokenizer& tokenizer, int max_length = 32)
        : model(model), tokenizer(tokenizer), max_length(max_length) {}
        
    std::string generate(const std::string& prompt) {
        try {
            std::cout << prompt;
            std::cout.flush();

            std::vector<int> prompt_tokens = tokenizer.encode(prompt);
            array input = array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, int32);
            
            const ModelArgs& args = model.get_args();
            int head_dim = args.hidden_size / args.num_attention_heads;
            std::vector<std::unique_ptr<KVCache>> cache;
            for (int i = 0; i < args.num_hidden_layers; i++) {
                cache.push_back(std::make_unique<KVCache>(args.num_key_value_heads, head_dim));
            }

            if (input.shape()[1] > 1) {
                std::vector<KVCache*> cache_ptrs;
                for (auto& c : cache) cache_ptrs.push_back(c.get());
                
                array context = slice(input, {0, 0}, {1, input.shape()[1] - 1});
                model.forward(context, &cache_ptrs);
                
                input = slice(input, {0, input.shape()[1] - 1}, {1, input.shape()[1]});
            }

            std::string generated_text;
            array y = input;

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
}; 