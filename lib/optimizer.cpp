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
    std::unordered_map<std::string, array> m;
    std::unordered_map<std::string, array> v;

public:
    AdamW(float lr = 1e-5, float b1 = 0.9f, float b2 = 0.97f, float eps = 1e-5f, float wd = 0.0f)
        : learning_rate(lr), beta1(b1), beta2(b2), eps(eps), weight_decay(wd), step(0) {}

    void update(Model& model, const std::unordered_map<std::string, array>& gradients) override {
        step++;
        auto& params = model.parameters();
        
        for (const auto& [name, grad] : gradients) {
            if (params.find(name) == params.end()) continue;
            
            // Initialize momentum arrays if they don't exist
            if (m.find(name) == m.end()) {
                m.emplace(name, zeros_like(grad));
                v.emplace(name, zeros_like(grad));
            }

            // Update biased first moment estimate
            array new_m = beta1 * m.at(name) + (1 - beta1) * grad;
            m.insert_or_assign(name, new_m);
            
            // Update biased second raw moment estimate
            array new_v = beta2 * v.at(name) + (1 - beta2) * square(grad);
            v.insert_or_assign(name, new_v);
            
            // Compute bias-corrected moments
            array m_hat = m.at(name) / (1 - std::pow(beta1, step));
            array v_hat = v.at(name) / (1 - std::pow(beta2, step));
            
            // Update parameters
            array param_update = learning_rate * m_hat / (sqrt(v_hat) + eps);
            array new_param = params.at(name) * (1 - learning_rate * weight_decay) - param_update;
            params.insert_or_assign(name, new_param);
        }
        
        model.set_parameters(params);
    }
}; 