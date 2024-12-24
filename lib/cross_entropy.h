#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <sstream>

using namespace mlx::core;

array cross_entropy(const array& logits, const array& targets, int axis = -1); 