#pragma once

#include <mlx/mlx.h>
#include <unordered_map>
#include <cmath>
#include "model.h"

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

    void update(Model& model, const std::unordered_map<std::string, array>& gradients) override;
}; 