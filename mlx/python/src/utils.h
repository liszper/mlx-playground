#pragma once
#include <numeric>
#include <optional>
#include <string>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/variant.h>

#include "mlx/array.h"

namespace nb = nanobind;

using namespace mlx::core;

using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<
    nb::bool_,
    nb::int_,
    nb::float_,
    array,
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>,
    std::complex<float>,
    nb::object>;

inline std::vector<int> get_reduce_axes(const IntOrVec& v, int dims) {
  std::vector<int> axes;
  if (std::holds_alternative<std::monostate>(v)) {
    axes.resize(dims);
    std::iota(axes.begin(), axes.end(), 0);
  } else if (auto pv = std::get_if<int>(&v); pv) {
    axes.push_back(*pv);
  } else {
    axes = std::get<std::vector<int>>(v);
  }
  return axes;
}

inline bool is_comparable_with_array(const ScalarOrArray& v) {
  if (auto pv = std::get_if<nb::object>(&v); pv) {
    return nb::isinstance<array>(*pv) || nb::hasattr(*pv, "__mlx_array__");
  } else {
    return true;
  }
}

inline nb::handle get_handle_of_object(const ScalarOrArray& v) {
  return std::get<nb::object>(v).ptr();
}

inline void throw_invalid_operation(
    const std::string& operation, const ScalarOrArray operand) {
  std::ostringstream msg;
  msg << "Cannot perform " << operation << " on an mlx.core.array and "
      << nb::type_name(get_handle_of_object(operand).type()).c_str();
  throw std::invalid_argument(msg.str());
}

array to_array(
    const ScalarOrArray& v,
    std::optional<Dtype> dtype = std::nullopt);

std::pair<array, array> to_arrays(
    const ScalarOrArray& a,
    const ScalarOrArray& b);

array to_array_with_accessor(nb::object obj);
