#pragma once

#include <variant>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"

namespace mlx::core {

using StreamOrDevice = std::variant<std::monostate, Stream, Device>;
Stream to_stream(StreamOrDevice s);

struct StreamContext {
 public:
  StreamContext(StreamOrDevice s) : _stream(default_stream(default_device())) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error("[StreamContext] Invalid argument, please specify a stream or device.");
    }
    auto _s = to_stream(s);
    set_default_device(_s.device);
    set_default_stream(_s);
  }

  ~StreamContext() {
    set_default_device(_stream.device);
    set_default_stream(_stream);
  }

 private:
  Stream _stream;
};

struct PrintFormatter {
  inline void print(std::ostream& os, bool val);
  inline void print(std::ostream& os, int16_t val);
  inline void print(std::ostream& os, uint16_t val);
  inline void print(std::ostream& os, int32_t val);
  inline void print(std::ostream& os, uint32_t val);
  inline void print(std::ostream& os, int64_t val);
  inline void print(std::ostream& os, uint64_t val);
  inline void print(std::ostream& os, float16_t val);
  inline void print(std::ostream& os, bfloat16_t val);
  inline void print(std::ostream& os, float val);

  bool capitalize_bool{false};
};

extern PrintFormatter global_formatter;

inline Dtype result_type(const array& a, const array& b) {
  return promote_types(a.dtype(), b.dtype());
}
inline Dtype result_type(const array& a, const array& b, const array& c) {
  return promote_types(result_type(a, b), c.dtype());
}
Dtype result_type(const std::vector<array>& arrays);

std::vector<int> broadcast_shapes(
    const std::vector<int>& s1,
    const std::vector<int>& s2);

bool is_same_shape(const std::vector<array>& arrays);

template <typename T>
int check_shape_dim(const T dim) {
  constexpr bool is_signed = std::numeric_limits<T>::is_signed;
  using U = std::conditional_t<is_signed, ssize_t, size_t>;
  constexpr U min = static_cast<U>(std::numeric_limits<int>::min());
  constexpr U max = static_cast<U>(std::numeric_limits<int>::max());

  if ((is_signed && dim < min) || dim > max) {
    throw std::invalid_argument("Shape dimension falls outside supported `int` range.");
  }

  return static_cast<int>(dim);
}

int normalize_axis(int axis, int ndim);

std::ostream& operator<<(std::ostream& os, const Device& d);
std::ostream& operator<<(std::ostream& os, const Stream& s);
std::ostream& operator<<(std::ostream& os, const Dtype& d);
std::ostream& operator<<(std::ostream& os, const Dtype::Kind& k);
std::ostream& operator<<(std::ostream& os, array a);
std::ostream& operator<<(std::ostream& os, const std::vector<int>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& v);
inline std::ostream& operator<<(std::ostream& os, const float16_t& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const bfloat16_t& v) {
  return os << static_cast<float>(v);
}

inline bool is_power_of_2(int n) {
  return ((n & (n - 1)) == 0) && n != 0;
}

inline int next_power_of_2(int n) {
  if (is_power_of_2(n)) {
    return n;
  }
  return pow(2, std::ceil(std::log2(n)));
}

}
