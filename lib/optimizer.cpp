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
    AdamW(float lr = 1e-5, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.01f)
        : learning_rate(lr), beta1(b1), beta2(b2), eps(eps), weight_decay(wd), step(0) {
        m.reserve(100);
        v.reserve(100);
    }

    void update(Model& model, const std::unordered_map<std::string, array>& gradients) override {
        step++;
        auto& params = model.parameters();
        
        const size_t BATCH_SIZE = 128;
        std::vector<array> arrays_to_eval;
        arrays_to_eval.reserve(BATCH_SIZE * 3);
        
        std::unordered_map<std::string, array> new_params;
        new_params.reserve(params.size());
        
        for (const auto& [name, grad] : gradients) {
            auto param_it = params.find(name);
            if (param_it == params.end()) continue;
            
            auto m_it = m.find(name);
            if (m_it == m.end()) {
                array m_init = zeros_like(grad);
                array v_init = zeros_like(grad);
                eval({m_init, v_init});
                m_it = m.emplace(name, std::move(m_init)).first;
                v.emplace(name, std::move(v_init));
            }
            
            array& param = param_it->second;
            array new_m = beta1 * m_it->second + (1.0f - beta1) * grad;
            array new_v = beta2 * v.at(name) + (1.0f - beta2) * square(grad);
            
            array param_update = learning_rate * new_m / (sqrt(new_v) + eps);
            array new_param = weight_decay > 0.0f 
                ? param * (1.0f - learning_rate * weight_decay) - param_update
                : param - param_update;
            
            arrays_to_eval.push_back(new_m);
            arrays_to_eval.push_back(new_v);
            arrays_to_eval.push_back(new_param);
            
            m.insert_or_assign(name, new_m);
            v.insert_or_assign(name, new_v);
            new_params.insert_or_assign(name, new_param);
            
            if (arrays_to_eval.size() >= BATCH_SIZE * 3) {
                eval(arrays_to_eval);
                arrays_to_eval.clear();
            }
        }
        
        if (!arrays_to_eval.empty()) {
            eval(arrays_to_eval);
        }
        
        model.set_parameters(new_params);
    }
}; 