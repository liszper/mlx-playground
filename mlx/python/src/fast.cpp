#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "python/src/utils.h"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_fast(nb::module_& parent_module) {
  auto m = parent_module.def_submodule("fast", "mlx.core.fast: fast operations");

  m.def("rms_norm",
      &fast::rms_norm,
      "x"_a,
      "weight"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def rms_norm(x: array, weight: array, eps: float, *, stream: Union[None, Stream, Device] = None) -> array"));

  m.def("rope",
      &fast::rope,
      "a"_a,
      "dims"_a,
      nb::kw_only(),
      "traditional"_a,
      "base"_a.none(),
      "scale"_a,
      "offset"_a,
      "freqs"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig("def rope(a: array, dims: int, *, traditional: bool, base: Optional[float], scale: float, offset: int, freqs: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"));

  m.def("scaled_dot_product_attention",
      &fast::scaled_dot_product_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "scale"_a,
      "mask"_a = nb::none(),
      "memory_efficient_threshold"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig("def scaled_dot_product_attention(q: array, k: array, v: array, *, scale: float,  mask: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array"));
}
