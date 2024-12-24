#include "mlx/backend/metal/metal.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");

  metal.def("is_available", &metal::is_available);
  metal.def("get_active_memory", &metal::get_active_memory);
  metal.def("get_peak_memory", &metal::get_peak_memory);
  metal.def("reset_peak_memory", &metal::reset_peak_memory);
  metal.def("get_cache_memory", &metal::get_cache_memory);
  metal.def("set_memory_limit", &metal::set_memory_limit, "limit"_a, nb::kw_only(), "relaxed"_a = true);
  metal.def("set_cache_limit", &metal::set_cache_limit, "limit"_a);
  metal.def("clear_cache", &metal::clear_cache);

  metal.def("start_capture", &metal::start_capture, "path"_a);
  metal.def("stop_capture", &metal::stop_capture);
  metal.def("device_info", &metal::device_info);
}
