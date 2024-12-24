#pragma once

#include <optional>

#include "mlx/array.h"

namespace mlx::core {

void async_eval(std::vector<array> outputs);

void eval(std::vector<array> outputs);

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
void eval(Arrays&&... outputs) {
  eval(std::vector<array>{std::forward<Arrays>(outputs)...});
}

std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotangents);

std::pair<array, array> vjp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& cotangent);

std::pair<std::vector<array>, std::vector<array>> jvp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& tangents);

std::pair<array, array> jvp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& tangent);

using ValueAndGradFn =
    std::function<std::pair<std::vector<array>, std::vector<array>>(
        const std::vector<array>&)>;
using SimpleValueAndGradFn = std::function<std::pair<array, std::vector<array>>(
    const std::vector<array>&)>;

ValueAndGradFn value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums);

ValueAndGradFn inline value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return value_and_grad(fun, std::vector<int>{argnum});
}

std::function<std::pair<array, array>(const array&)> inline value_and_grad(
    const std::function<array(const array&)>& fun) {
  return [fun](auto inputs) { return vjp(fun, inputs, array(1.0f)); };
}

SimpleValueAndGradFn inline value_and_grad(
    const std::function<array(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  return [fun, argnums](auto inputs) {
    auto result = value_and_grad(
        [fun](auto inputs) { return std::vector<array>{fun(inputs)}; },
        argnums)(inputs);

    return std::make_pair(result.first[0], result.second);
  };
}

SimpleValueAndGradFn inline value_and_grad(
    const std::function<array(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return value_and_grad(fun, std::vector<int>{argnum});
}

std::function<std::vector<array>(const std::vector<array>&)> inline grad(
    const std::function<array(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  auto fn = value_and_grad(fun, argnums);
  return [fn](const std::vector<array>& inputs) { return fn(inputs).second; };
}

std::function<std::vector<array>(const std::vector<array>&)> inline grad(
    const std::function<array(const std::vector<array>&)>& fun,
    int argnum = 0) {
  return grad(fun, std::vector<int>{argnum});
}

std::function<array(const array&)> inline grad(
    const std::function<array(const array&)>& fun) {
  auto fn = value_and_grad(fun);
  return [fn](const array& input) { return fn(input).second; };
}

}
