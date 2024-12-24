#include "cross_entropy.h"

array cross_entropy(const array& logits, const array& targets, int axis) {
    bool targets_as_probs = targets.ndim() == logits.ndim();
    
    // Helper function to drop dimension at specified axis
    auto drop_dim = [](const std::vector<int>& shape, int axis) {
        std::vector<int> new_shape = shape;
        if (axis < 0) axis += shape.size();
        new_shape.erase(new_shape.begin() + axis);
        return new_shape;
    };
    
    // Get shapes for validation
    std::vector<int> logits_shape = logits.shape();
    std::vector<int> targets_shape = targets.shape();
    std::vector<int> expected_shape = drop_dim(logits_shape, axis);
    
    // Validate shapes
    if (targets_as_probs) {
        if (targets_shape != logits_shape) {
            std::stringstream err;
            err << "Targets shape [";
            for (size_t i = 0; i < targets_shape.size(); ++i) {
                err << targets_shape[i];
                if (i < targets_shape.size() - 1) err << ", ";
            }
            err << "] does not match logits shape [";
            for (size_t i = 0; i < logits_shape.size(); ++i) {
                err << logits_shape[i];
                if (i < logits_shape.size() - 1) err << ", ";
            }
            err << "]";
            throw std::runtime_error(err.str());
        }
        array score = sum(logits * targets, axis);
        return logsumexp(logits, axis) - score;
    } else {
        if (targets_shape != expected_shape) {
            std::stringstream err;
            err << "Sparse targets shape [";
            for (size_t i = 0; i < targets_shape.size(); ++i) {
                err << targets_shape[i];
                if (i < targets_shape.size() - 1) err << ", ";
            }
            err << "] does not match expected shape [";
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                err << expected_shape[i];
                if (i < expected_shape.size() - 1) err << ", ";
            }
            err << "] for logits shape [";
            for (size_t i = 0; i < logits_shape.size(); ++i) {
                err << logits_shape[i];
                if (i < logits_shape.size() - 1) err << ", ";
            }
            err << "]";
            throw std::runtime_error(err.str());
        }
        
        // Handle sparse targets (indices)
        array score = reshape(take_along_axis(logits, expand_dims(targets, -1), axis), 
                            targets_shape);
        return logsumexp(logits, axis) - score;
    }
} 