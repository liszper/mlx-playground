#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <chrono>

#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/random.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;
using namespace mlx::core::random;

class PyKeySequence {
 public:
  explicit PyKeySequence(uint64_t seed) {
    state_.append(key(seed));
  }

  void seed(uint64_t seed) {
    state_[0] = key(seed);
  }

  array next() {
    auto out = split(nb::cast<array>(state_[0]));
    state_[0] = out.first;
    return out.second;
  }

  nb::list state() {
    return state_;
  }

  void release() {
    nb::gil_scoped_acquire gil;
    state_.release().dec_ref();
  }

 private:
  nb::list state_;
};

PyKeySequence& default_key() {
  auto get_current_time_seed = []() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
  };
  static PyKeySequence ks(get_current_time_seed());
  return ks;
}

void init_random(nb::module_& parent_module) {
  auto m = parent_module.def_submodule("random",
      "mlx.core.random: functionality related to random number generation");

  m.attr("state") = default_key().state();
  m.def("seed",
      [](uint64_t seed) { default_key().seed(seed); },
      "seed"_a);
  m.def("key",
      &key,
      "seed"_a);
  m.def("split",
      nb::overload_cast<const array&, int, StreamOrDevice>(&random::split),
      "key"_a,
      "num"_a = 2,
      "stream"_a = nb::none(),
      nb::sig("def split(key: array, num: int = 2, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("uniform", [](const ScalarOrArray& low, const ScalarOrArray& high, const std::vector<int>& shape, std::optional<Dtype> type, const std::optional<array>& key_, StreamOrDevice s) {
        auto key = key_ ? key_.value() : default_key().next();
        return uniform(
            to_array(low),
            to_array(high),
            shape,
            type.value_or(float32),
            key,
            s);
      },
      "low"_a = 0,
      "high"_a = 1,
      "shape"_a = std::vector<int>{},
      "dtype"_a.none() = float32,
      "key"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig("def uniform(low: Union[scalar, array] = 0, high: Union[scalar, array] = 1, shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"));
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() { default_key().release(); }));
}
