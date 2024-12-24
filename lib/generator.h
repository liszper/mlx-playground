#pragma once

#include <string>
#include "model.h"
#include "tokenizer.h"
#include "model_args.h"
#include "transformer.h"

class Generator {
private:
    TransformerModel& model;
    Tokenizer& tokenizer;
    int max_length;
    
public:
    Generator(TransformerModel& model, Tokenizer& tokenizer, int max_length = 256);
    std::string generate(const std::string& prompt);
}; 