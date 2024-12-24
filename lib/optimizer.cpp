#include "optimizer.h"

void AdamW::update(Model& model, const std::unordered_map<std::string, array>& gradients) {
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