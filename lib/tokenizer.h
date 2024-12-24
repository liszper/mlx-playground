#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    int eos_token_id = 128001;  // <|end_of_text|>
    int bos_token_id = 128000;  // <|begin_of_text|>

    Tokenizer(const std::string& model_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> decoder;
    
    std::vector<std::string> tokenize_text(const std::string& text);
    static std::string base64_decode(const std::string& encoded);
    static int get_utf8_char_length(unsigned char c);
}; 