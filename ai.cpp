#include <mlx/mlx.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <optional>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <iomanip>
#include <limits>

#include "lib/kv_cache.cpp"
#include "lib/rms_norm.cpp"
#include "lib/linear.cpp"
#include "lib/rope.cpp"
#include "lib/embedding.cpp"
#include "lib/cross_entropy.cpp"
#include "lib/model_args.cpp"
#include "lib/feed_forward.cpp"
#include "lib/model.cpp"
#include "lib/optimizer.cpp"
#include "lib/transformer.cpp"
#include "lib/tokenizer.cpp"
#include "lib/generator.cpp"

using namespace mlx::core;

// Example usage in main
int main() {
    try {
        std::cout << "\n=== MLX C++ Language Model Training ===\n" << std::endl;

        // Initialize model and tokenizer
        std::cout << "🔧 Initializing model and tokenizer..." << std::endl;
        TransformerModel model;
        Tokenizer tokenizer("./tokenizer.model");
        Generator generator(model, tokenizer);
        std::cout << "✓ Initialization complete\n" << std::endl;

        // First generation test
        std::string prompt = "Már nem volt fiatal, de még";
        std::cout << "📝 Initial generation test with prompt:\n\"" << prompt << "\"\n" << std::endl;
        std::cout << "Generated text:" << std::endl;
        std::cout << "----------------" << std::endl;
        generator.generate(prompt);
        std::cout << "----------------\n" << std::endl;

        // Training preparation
        std::cout << "🔄 Preparing training data..." << std::endl;
        std::string training_text = "Már nem volt fiatal, de még elég jól bírta magát; "
            "ismerték és félték a nádas lakói, de még azon túl is, közelben-távolban, "
            "minden négylábú lény. Látása nem romlott, s ha ezerméteres magasságból "
            "kiszemelte zsákmányát, úgy csapott le rá, mint egy kalapács, mely egyetlen "
            "ütéssel veri be a szöget. És így, viruló korában, ereje teljében, két lassú "
            "szárnycsapás között egyszer csak megállt a szíve. De nem mertek előbújni sem "
            "a nyulak, sem az ürgék, sem a környező falvak baromfiai, mert ő ott lebegett "
            "ezer méter magasban, kiterjesztett szárnyával, fenyegető mozdulatlanságban "
            "túlélve a halált még két vagy három perccel, míg el nem állt a szél.";

        std::vector<int> tokens = tokenizer.encode(training_text);
        
        // Create input/target arrays
        std::vector<int> token_vec(tokens.begin(), tokens.end() - 1);
        array x = array(token_vec.data(), {1, (int)token_vec.size()}, int32);

        std::vector<int> target_vec(tokens.begin() + 1, tokens.end());
        array y = array(target_vec.data(), {1, (int)target_vec.size()}, int32);

        array z = array({(int)tokens.size()}, {1}, int32);
        
        std::cout << "✓ Training data prepared" << std::endl;
        std::cout << "  • Sequence length: " << tokens.size() << " tokens" << std::endl;
        std::cout << "  • Input shape: [1, " << token_vec.size() << "]" << std::endl;
        std::cout << "  • Target shape: [1, " << target_vec.size() << "]\n" << std::endl;

        // Initialize optimizer
        std::cout << "🔧 Initializing AdamW optimizer with:" << std::endl;
        std::cout << "  • Learning rate: 1e-5" << std::endl;
        std::cout << "  • Beta1: 0.9" << std::endl;
        std::cout << "  • Beta2: 0.97" << std::endl;
        std::cout << "  • Epsilon: 1e-5" << std::endl;
        std::cout << "  • Weight decay: 0.0\n" << std::endl;
        
        std::unique_ptr<AdamW> optimizer = std::make_unique<AdamW>(
            1e-5, 0.9, 0.97, 1e-5, 0.0
        );

        // Training loop
        std::cout << "🚀 Starting training loop (20 iterations)" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        float best_loss = std::numeric_limits<float>::max();
        
        // Pre-allocate vectors and reuse them
        std::vector<array> state_arrays;
        state_arrays.reserve(1000);  // Reserve space for arrays
        
        // Cache model states to avoid recomputation
        auto model_states = model.state_arrays();
        state_arrays.insert(state_arrays.end(), model_states.begin(), model_states.end());

        for (int step = 0; step < 20; ++step) {
            // Clear previous state arrays but maintain capacity
            state_arrays.clear();
            
            // Get loss and gradients in a single pass
            auto [loss, grads] = model.value_and_grad(x, y, z);
            
            // Update model parameters immediately
            optimizer->update(model, grads);
            
            // Add loss and model states to evaluation batch
            state_arrays.push_back(loss);
            state_arrays.insert(state_arrays.end(), model_states.begin(), model_states.end());
            
            // Add gradient arrays to evaluation batch
            for (const auto& [_, param] : grads) {
                state_arrays.push_back(param);
            }
            
            // Evaluate all arrays at once
            eval(state_arrays);
            
            float loss_value = loss.item<float>();
            best_loss = std::min(best_loss, loss_value);
            
            std::cout << "Step " << std::setw(2) << step + 1 << "/20"
                     << " | Loss: " << std::fixed << std::setprecision(6) << loss_value;
            
            if (loss_value == best_loss) {
                std::cout << " (best) ⭐";
            }
            std::cout << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "✓ Training complete" << std::endl;
        std::cout << "  • Final loss: " << std::fixed << std::setprecision(6) << best_loss << "\n" << std::endl;

        // Final generation test
        std::cout << "📝 Final generation test with same prompt:\n\"" << prompt << "\"\n" << std::endl;
        std::cout << "Generated text:" << std::endl;
        std::cout << "----------------" << std::endl;
        generator.generate(prompt);
        std::cout << "----------------\n" << std::endl;

        std::cout << "✨ All operations completed successfully!\n" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
} 