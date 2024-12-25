#pragma once

#include <mlx/mlx.h>
#include <unordered_map>
#include <cmath>
#include "model.cpp"

using namespace mlx::core;

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Model& model, const std::unordered_map<std::string, array>& gradients) = 0;
};

class AdamW : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step;
    std::unordered_map<std::string, std::optional<array>> m;
    std::unordered_map<std::string, std::optional<array>> v;

public:
    AdamW(float lr = 1e-5, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), eps(eps), weight_decay(wd), step(0) {
        m.reserve(100);
        v.reserve(100);
    }

    void update(Model& model, const std::unordered_map<std::string, array>& gradients) override {
        step++;
        auto& params = model.parameters();
        
        // Pre-allocate all arrays we'll need
        std::vector<array> all_updates;
        all_updates.reserve(gradients.size() * 3);  // m, v, and param updates
        
        // Use optional for new_params as well
        std::unordered_map<std::string, std::optional<array>> new_params;
        new_params.reserve(params.size());
        
        // Batch all operations
        for (const auto& [name, grad] : gradients) {
            auto param_it = params.find(name);
            if (param_it == params.end()) continue;
            
            // Initialize momentum arrays if needed
            auto m_it = m.find(name);
            if (m_it == m.end() || !m_it->second.has_value()) {
                m[name] = zeros_like(grad);
                v[name] = zeros_like(grad);
            }
            
            // Compute all updates in one go
            array& param = param_it->second;
            array& m_val = m[name].value();
            array& v_val = v[name].value();
            
            // Use MLX's stream operations to chain computations
            auto new_m = beta1 * m_val + (1.0f - beta1) * grad;
            auto new_v = beta2 * v_val + (1.0f - beta2) * square(grad);
            auto param_update = learning_rate * new_m / (sqrt(new_v) + eps);
            auto new_param = weight_decay > 0.0f 
                ? param * (1.0f - learning_rate * weight_decay) - param_update
                : param - param_update;
            
            // Store updates
            m[name] = new_m;
            v[name] = new_v;
            new_params[name] = new_param;
            
            // Add to evaluation batch
            all_updates.push_back(new_m);
            all_updates.push_back(new_v);
            all_updates.push_back(new_param);
        }
        
        // Single evaluation for all updates
        eval(all_updates);
        
        // Convert new_params back to regular map before updating model
        std::unordered_map<std::string, array> final_params;
        final_params.reserve(new_params.size());
        for (const auto& [name, param_opt] : new_params) {
            if (param_opt.has_value()) {
                final_params.emplace(name, param_opt.value());
            }
        }
        
        // Update model parameters
        model.set_parameters(final_params);
    }
}; 