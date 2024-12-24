#include <cmath>
#include <sstream>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/utils.h"

namespace mlx::core::random {

KeySequence::KeySequence(uint64_t seed) : key_(key(seed)) {}

void KeySequence::seed(uint64_t seed) {
  key_ = key((seed));
}

array KeySequence::next() {
  auto out = split(key_);
  key_ = out.first;
  return out.second;
}

void seed(uint64_t seed) {
  KeySequence::default_().seed(seed);
}

array key(uint64_t seed) {
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  uint32_t k2 = static_cast<uint32_t>(seed);
  return array({k1, k2});
}

array bits(
    const std::vector<int>& shape,
    int width /* 4 */,
    const std::optional<array>& key_ /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  auto key = key_ ? *key_ : KeySequence::default_().next();
  if (key.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[bits] Expected key type uint32 but received " << key.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (key.shape() != std::vector<int>{2}) {
    std::ostringstream msg;
    msg << "[bits] Expected key shape (2) but received " << key.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto get_dtype = [width]() {
    switch (width) {
      case 4:
        return uint32;
      case 2:
        return uint16;
      case 1:
        return uint8;
      default:
        std::ostringstream msg;
        msg << "[bits] Bit width must be in {1, 2, 4} but got " << width << ".";
        throw std::invalid_argument(msg.str());
    }
  };
  return array(
      shape,
      get_dtype(),
      std::make_shared<RandomBits>(to_stream(s), shape, width),
      {key});
}

std::pair<array, array> split(const array& key, StreamOrDevice s /* = {} */) {
  auto stream = to_stream(s);
  auto out = mlx::core::split(random::split(key, 2, stream), 2, stream);
  return {reshape(out[0], {2}, stream), reshape(out[1], {2}, stream)};
}

array split(const array& key, int num, StreamOrDevice s /* = {} */) {
  return bits({num, 2}, 4, key, s);
}

template <typename T>
T below_one() {
  T f = T(1.0);
  uint16_t* m = (uint16_t*)&f;
  *m -= 1;
  return f;
}

template <typename T>
T above_minus_one() {
  T f = T(-1.0);
  uint16_t* m = (uint16_t*)&f;
  *m -= 1;
  return f;
}

array above_minus_one_with_default(Dtype dtype) {
  switch (dtype) {
    case float16:
      return array(above_minus_one<float16_t>(), dtype);
    case bfloat16:
      return array(above_minus_one<bfloat16_t>(), dtype);
    default:
      return array(std::nextafter(-1.0f, 0.0f), dtype);
  }
}

array uniform(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!issubdtype(dtype, floating)) {
    throw std::invalid_argument("[uniform] Can only generate uniform numbers with real "
        "floating point type.");
  }

  auto stream = to_stream(s);
  auto lo = astype(low, dtype, stream);
  auto hi = astype(high, dtype, stream);
  auto range = subtract(hi, lo, stream);
  auto out_shape = broadcast_shapes(shape, range.shape());
  if (out_shape != shape) {
    std::ostringstream msg;
    msg << "[uniform] Cannot generate random values of shape " << shape
        << " from broadcasted shape " << out_shape << ".";
    throw std::invalid_argument(msg.str());
  }
  auto get_limits = [&dtype]() {
    switch (dtype) {
      case float32:
        return std::make_pair(
            array(std::nextafter(1.0f, 0.0f), float32),
            array(std::numeric_limits<uint32_t>::max(), float32));
      case float16:
        return std::make_pair(
            array(below_one<float16_t>(), float16),
            array(std::numeric_limits<uint16_t>::max(), float32));
      case bfloat16:
        return std::make_pair(
            array(below_one<bfloat16_t>(), bfloat16),
            array(std::numeric_limits<uint16_t>::max(), float32));
      default:
        throw std::runtime_error("[uniform] Unsupported type.");
    }
  };

  auto [upper, maxval] = get_limits();
  auto out = bits(shape, size_of(dtype), key, stream);
  out = astype(divide(out, maxval, stream), dtype, stream);
  out = minimum(out, upper, stream);
  return add(multiply(range, out, stream), lo, stream);
}

array uniform(
    const std::vector<int>& shape,
    Dtype dtype,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  return uniform(
      array(0.0, dtype), array(1.0, dtype), shape, dtype, key, to_stream(s));
}

}
