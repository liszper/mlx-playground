#pragma once

#include <chrono>
#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::random {

class KeySequence {
 public:
  explicit KeySequence(uint64_t seed);

  void seed(uint64_t seed);
  array next();

  static KeySequence& default_() {
    static KeySequence ks(get_current_time_seed());
    return ks;
  }

 private:
  array key_;
  static uint64_t get_current_time_seed() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
  }
};

array key(uint64_t seed);

void seed(uint64_t seed);

array bits(
    const std::vector<int>& shape,
    int width,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array bits(
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bits(shape, 4, key, s);
}

std::pair<array, array> split(const array& key, StreamOrDevice s = {});

array split(const array& key, int num, StreamOrDevice s = {});

array uniform(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T, typename U>
array uniform(
    T low,
    U high,
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(array(low), array(high), shape, dtype, key, to_stream(s));
}

array uniform(
    const std::vector<int>& shape,
    Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array uniform(
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(shape, float32, key);
}

}
