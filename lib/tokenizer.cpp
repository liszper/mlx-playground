#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

class Tokenizer {
public:
    int eos_token_id = 128001;  // <|end_of_text|>
    int bos_token_id = 128000;  // <|begin_of_text|>

    Tokenizer(const std::string& model_path) {
        std::ifstream file(model_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open tokenizer model file: " + model_path);
        }
        
        std::string line;
        int count = 0;
        while (std::getline(file, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string token = line.substr(0, space_pos);
                int rank = std::stoi(line.substr(space_pos + 1));
                
                std::string decoded = base64_decode(token);
                vocab[decoded] = rank;
                decoder[rank] = decoded;
                count++;
            }
        }
        
        std::cout << "Loaded " << count << " tokens from vocabulary" << std::endl;
        
        // Add special tokens if they don't exist
        if (vocab.find("<|endoftext|>") == vocab.end()) {
            vocab["<|endoftext|>"] = eos_token_id;
            decoder[eos_token_id] = "<|endoftext|>";
        }
        if (vocab.find("<|startoftext|>") == vocab.end()) {
            vocab["<|startoftext|>"] = bos_token_id;
            decoder[bos_token_id] = "<|startoftext|>";
        }
    }

    std::vector<int> encode(const std::string& text) {
        if (text.empty()) {
            std::cerr << "Warning: Empty input text" << std::endl;
            return {};
        }
        
        std::vector<int> tokens;
        auto text_tokens = tokenize_text(text);
        
        for (const auto& token : text_tokens) {
            if (vocab.find(token) != vocab.end()) {
                tokens.push_back(vocab[token]);
            } else {
                // Handle unknown tokens by encoding each character separately
                for (size_t i = 0; i < token.length();) {
                    int char_len = get_utf8_char_length(token[i]);
                    std::string utf8_char = token.substr(i, char_len);
                    
                    if (vocab.find(utf8_char) != vocab.end()) {
                        tokens.push_back(vocab[utf8_char]);
                    } else {
                        // If character not found, use a special unknown token or skip
                        std::cerr << "Warning: Unknown token: " << utf8_char << std::endl;
                    }
                    
                    i += char_len;
                }
            }
        }
        
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int token : tokens) {
            if (decoder.find(token) != decoder.end()) {
                result += decoder[token];
            } else {
                std::cerr << "Warning: Unknown token ID: " << token << std::endl;
            }
        }
        return result;
    }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> decoder;
    
    std::vector<std::string> tokenize_text(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current = "";
        
        for (size_t i = 0; i < text.length();) {
            int char_len = get_utf8_char_length(text[i]);
            if (char_len == 0) break;
            
            std::string utf8_char = text.substr(i, char_len);
            
            // Handle whitespace
            if (utf8_char == " " || utf8_char == "\n" || utf8_char == "\t") {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current = "";
                }
                tokens.push_back(utf8_char);
            } else {
                current += utf8_char;
            }
            
            i += char_len;
        }
        
        if (!current.empty()) {
            tokens.push_back(current);
        }
        
        return tokens;
    }

    static std::string base64_decode(const std::string& encoded) {
        static const std::string base64_chars = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            
        std::string decoded;
        std::vector<int> vec;
        int val = 0;
        int bits = -8;
        
        for (char c : encoded) {
            if (c == '=') break;
            
            size_t pos = base64_chars.find(c);
            if (pos != std::string::npos) {
                val = (val << 6) + static_cast<int>(pos);
                bits += 6;
                
                if (bits >= 0) {
                    decoded.push_back(static_cast<char>((val >> bits) & 0xFF));
                    bits -= 8;
                }
            }
        }
        
        return decoded;
    }

    static int get_utf8_char_length(unsigned char c) {
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 0;  // Invalid UTF-8 character
    }
}; 