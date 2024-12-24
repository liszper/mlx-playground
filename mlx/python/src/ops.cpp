#include <numeric>
#include <ostream>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

using Scalar = std::variant<int, double>;

Dtype scalar_to_dtype(Scalar scalar) {
  if (std::holds_alternative<int>(scalar)) {
    return int32;
  } else {
    return float32;
  }
}

double scalar_to_double(Scalar s) {
  if (std::holds_alternative<double>(s)) {
    return std::get<double>(s);
  } else {
    return static_cast<double>(std::get<int>(s));
  }
}

void init_ops(nb::module_& m) {
  m.def("reshape",
      &reshape,
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def reshape(a: array, /, shape: Sequence[int], *, stream: "
              "Union[None, Stream, Device] = None) -> array"));
  m.def("flatten",
      [](const array& a,
         int start_axis,
         int end_axis,
         const StreamOrDevice& s) { return flatten(a, start_axis, end_axis); },
      nb::arg(),
      "start_axis"_a = 0,
      "end_axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def flatten(a: array, /, start_axis: int = 0, end_axis: int = "
              "-1, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("squeeze", [](const array& a, const IntOrVec& v, const StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return squeeze(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return squeeze(a, *pv, s);
        } else {
          return squeeze(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def squeeze(a: array, /, axis: Union[None, int, Sequence[int]] = "
          "None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("expand_dims", [](const array& a, const std::variant<int, std::vector<int>>& v, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&v); pv) {
          return expand_dims(a, *pv, s);
        } else {
          return expand_dims(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def expand_dims(a: array, /, axis: Union[int, Sequence[int]], "
              "*, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("abs", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::abs(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def abs(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sign", [](const ScalarOrArray& a, StreamOrDevice s) {
        return sign(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sign(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("negative", [](const ScalarOrArray& a, StreamOrDevice s) {
        return negative(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def negative(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("add", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return add(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def add(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("subtract", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return subtract(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def subtract(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("multiply", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return multiply(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def multiply(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("divide", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("divmod", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return divmod(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def divmod(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("floor_divide", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return floor_divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def floor_divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("remainder", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return remainder(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def remainder(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("equal", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("not_equal", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return not_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def not_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("less", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return less(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def less(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("less_equal", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return less_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def less_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("greater", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return greater(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def greater(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("greater_equal", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return greater_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def greater_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("array_equal", [](const ScalarOrArray& a_, const ScalarOrArray& b_, bool equal_nan, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return array_equal(a, b, equal_nan, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig("def array_equal(a: Union[scalar, array], b: Union[scalar, array], equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("matmul",
      &matmul,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def matmul(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("square", [](const ScalarOrArray& a, StreamOrDevice s) {
        return square(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def square(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sqrt", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::sqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("rsqrt", [](const ScalarOrArray& a, StreamOrDevice s) {
        return rsqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def rsqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("reciprocal", [](const ScalarOrArray& a, StreamOrDevice s) {
        return reciprocal(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def reciprocal(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("logical_not", [](const ScalarOrArray& a, StreamOrDevice s) {
        return logical_not(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def logical_not(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("logical_and", [](const ScalarOrArray& a, const ScalarOrArray& b, StreamOrDevice s) {
        return logical_and(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def logical_and(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));

  m.def("logical_or", [](const ScalarOrArray& a, const ScalarOrArray& b, StreamOrDevice s) {
        return logical_or(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def logical_or(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("exp", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::exp(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def exp(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("expm1", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::expm1(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def expm1(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("erf", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::erf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def erf(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("erfinv", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::erfinv(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def erfinv(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sin", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::sin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cos", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::cos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def cos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("tan", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::tan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def tan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sinh", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::sinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cosh", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::cosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def cosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("tanh", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::tanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def tanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("degrees", [](const ScalarOrArray& a, StreamOrDevice s) {
        return degrees(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def degrees(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("radians", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::radians(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def radians(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("log", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::log(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def log(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("log2", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::log2(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def log2(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("log10", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::log10(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def log10(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("log1p", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::log1p(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def log1p(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("stop_gradient",
      &stop_gradient,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def stop_gradient(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sigmoid", [](const ScalarOrArray& a, StreamOrDevice s) {
        return sigmoid(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("power", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return power(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def power(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("arange", [](Scalar start, Scalar stop, const std::optional<Scalar>& step, const std::optional<Dtype>& dtype_, StreamOrDevice s) {
        Dtype dtype = dtype_
            ? *dtype_
            : promote_types(
                  scalar_to_dtype(start),
                  step ? promote_types(
                             scalar_to_dtype(stop), scalar_to_dtype(*step))
                       : scalar_to_dtype(stop));
        return arange(
            scalar_to_double(start),
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "start"_a,
      "stop"_a,
      "step"_a = nb::none(),
      nb::kw_only(),
      "dtype"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig("def arange(start : Union[int, float], stop : Union[int, float], step : Union[None, int, float], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("arange", [](Scalar stop, const std::optional<Scalar>& step, const std::optional<Dtype>& dtype_, StreamOrDevice s) {
        Dtype dtype = dtype_ ? *dtype_
            : step
            ? promote_types(scalar_to_dtype(stop), scalar_to_dtype(*step))
            : scalar_to_dtype(stop);
        return arange(
            0.0,
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "stop"_a,
      "step"_a = nb::none(),
      nb::kw_only(),
      "dtype"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig("def arange(stop : Union[int, float], step : Union[None, int, float] = None, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("linspace", [](Scalar start, Scalar stop, int num, std::optional<Dtype> dtype, StreamOrDevice s) {
        return linspace(
            scalar_to_double(start),
            scalar_to_double(stop),
            num,
            dtype.value_or(float32),
            s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "dtype"_a.none() = float32,
      "stream"_a = nb::none(),
      nb::sig("def linspace(start, stop, num: Optional[int] = 50, dtype: Optional[Dtype] = float32, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("take", [](const array& a, const std::variant<int, array>& indices, const std::optional<int>& axis, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices); pv) {
          return axis ? take(a, *pv, axis.value(), s) : take(a, *pv, s);
        } else {
          auto indices_ = std::get<array>(indices);
          return axis ? take(a, indices_, axis.value(), s)
                      : take(a, indices_, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def take(a: array, /, indices: Union[int, array], axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("take_along_axis", [](const array& a, const array& indices, const std::optional<int>& axis, StreamOrDevice s) {
        if (axis.has_value()) {
          return take_along_axis(a, indices, axis.value(), s);
        } else {
          return take_along_axis(reshape(a, {-1}, s), indices, 0, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def take_along_axis(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("put_along_axis", [](const array& a, const array& indices, const array& values, const std::optional<int>& axis, StreamOrDevice s) {
        if (axis.has_value()) {
          return put_along_axis(a, indices, values, axis.value(), s);
        } else {
          return reshape(
              put_along_axis(reshape(a, {-1}, s), indices, values, 0, s),
              a.shape(),
              s);
        }
      },
      nb::arg(),
      "indices"_a,
      "values"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def put_along_axis(a: array, /, indices: array, values: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("full", [](const std::variant<int, std::vector<int>>& shape, const ScalarOrArray& vals, std::optional<Dtype> dtype, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&shape); pv) {
          return full({*pv}, to_array(vals, dtype), s);
        } else {
          return full(
              std::get<std::vector<int>>(shape), to_array(vals, dtype), s);
        }
      },
      "shape"_a,
      "vals"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def full(shape: Union[int, Sequence[int]], vals: Union[scalar, array], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("zeros", [](const std::variant<int, std::vector<int>>& shape, std::optional<Dtype> dtype, StreamOrDevice s) {
        auto t = dtype.value_or(float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return zeros({*pv}, t, s);
        } else {
          return zeros(std::get<std::vector<int>>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def zeros(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("zeros_like",
      &zeros_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def zeros_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("ones", [](const std::variant<int, std::vector<int>>& shape, std::optional<Dtype> dtype, StreamOrDevice s) {
        auto t = dtype.value_or(float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return ones({*pv}, t, s);
        } else {
          return ones(std::get<std::vector<int>>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def ones(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("ones_like",
      &ones_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def ones_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("eye", [](int n, std::optional<int> m, int k, std::optional<Dtype> dtype, StreamOrDevice s) {
        return eye(n, m.value_or(n), k, dtype.value_or(float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def eye(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("identity", [](int n, std::optional<Dtype> dtype, StreamOrDevice s) {
        return identity(n, dtype.value_or(float32), s);
      },
      "n"_a,
      "dtype"_a.none() = float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def identity(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("tri", [](int n, std::optional<int> m, int k, std::optional<Dtype> type, StreamOrDevice s) {
        return tri(n, m.value_or(n), k, type.value_or(float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def tri(n: int, m: int, k: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("tril",
      &tril,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def tril(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("triu",
      &triu,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def triu(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("allclose",
      &allclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig("def allclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isclose",
      &isclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig("def isclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("all", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def all(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("any", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def any(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("minimum", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return minimum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def minimum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("maximum", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return maximum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def maximum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("floor", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::floor(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def floor(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("ceil", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::ceil(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def ceil(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isnan", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::isnan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isnan(a: array, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isinf", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::isinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isinf(a: array, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isfinite", [](const ScalarOrArray& a, StreamOrDevice s) {
        return mlx::core::isfinite(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isfinite(a: array, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isposinf", [](const ScalarOrArray& a, StreamOrDevice s) {
        return isposinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isposinf(a: array, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("isneginf", [](const ScalarOrArray& a, StreamOrDevice s) {
        return isneginf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def isneginf(a: array, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("moveaxis",
      &moveaxis,
      nb::arg(),
      "source"_a,
      "destination"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def moveaxis(a: array, /, source: int, destination: int, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("swapaxes",
      &swapaxes,
      nb::arg(),
      "axis1"_a,
      "axis2"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def swapaxes(a: array, /, axis1 : int, axis2: int, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("transpose", [](const array& a, const std::optional<std::vector<int>>& axes, StreamOrDevice s) {
        if (axes.has_value()) {
          return transpose(a, *axes, s);
        } else {
          return transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def transpose(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("permute_dims", [](const array& a, const std::optional<std::vector<int>>& axes, StreamOrDevice s) {
        if (axes.has_value()) {
          return transpose(a, *axes, s);
        } else {
          return transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def permute_dims(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("sum", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      "array"_a,
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def sum(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("prod", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def prod(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("min", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def min(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("max", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def max(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("logsumexp", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def logsumexp(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("mean", [](const array& a, const IntOrVec& axis, bool keepdims, StreamOrDevice s) {
        return mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def mean(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("var", [](const array& a, const IntOrVec& axis, bool keepdims, int ddof, StreamOrDevice s) {
        return var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def var(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("std", [](const array& a, const IntOrVec& axis, bool keepdims, int ddof, StreamOrDevice s) {
        return mlx::core::std(
            a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def std(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("split", [](const array& a, const std::variant<int, std::vector<int>>& indices_or_sections, int axis, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
          return split(a, *pv, axis, s);
        } else {
          return split(
              a, std::get<std::vector<int>>(indices_or_sections), axis, s);
        }
      },
      nb::arg(),
      "indices_or_sections"_a,
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def split(a: array, /, indices_or_sections: Union[int, Sequence[int]], axis: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("argmin", [](const array& a, std::optional<int> axis, bool keepdims, StreamOrDevice s) {
        if (axis) {
          return argmin(a, *axis, keepdims, s);
        } else {
          return argmin(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def argmin(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("argmax", [](const array& a, std::optional<int> axis, bool keepdims, StreamOrDevice s) {
        if (axis) {
          return argmax(a, *axis, keepdims, s);
        } else {
          return argmax(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def argmax(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("broadcast_to",
      [](const ScalarOrArray& a,
         const std::vector<int>& shape,
         StreamOrDevice s) { return broadcast_to(to_array(a), shape, s); },
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def broadcast_to(a: Union[scalar, array], /, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("softmax", [](const array& a, const IntOrVec& axis, bool precise, StreamOrDevice s) {
        return softmax(a, get_reduce_axes(axis, a.ndim()), precise, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "precise"_a = false,
      "stream"_a = nb::none(),
      nb::sig("def softmax(a: array, /, axis: Union[None, int, Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("concatenate", [](const std::vector<array>& arrays, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return concatenate(arrays, *axis, s);
        } else {
          return concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def concatenate(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("concat", [](const std::vector<array>& arrays, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return concatenate(arrays, *axis, s);
        } else {
          return concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def concat(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("stack", [](const std::vector<array>& arrays, std::optional<int> axis, StreamOrDevice s) {
        if (axis.has_value()) {
          return stack(arrays, axis.value(), s);
        } else {
          return stack(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def stack(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("meshgrid", [](nb::args arrays_, bool sparse, std::string indexing, StreamOrDevice s) {
        std::vector<array> arrays = nb::cast<std::vector<array>>(arrays_);
        return meshgrid(arrays, sparse, indexing, s);
      },
      "arrays"_a,
      "sparse"_a = false,
      "indexing"_a = "xy",
      "stream"_a = nb::none(),
      nb::sig("def meshgrid(*arrays: array, sparse: Optional[bool] = False, indexing: Optional[str] = 'xy', stream: Union[None, Stream, Device] = None) -> array"));
  m.def("repeat", [](const array& array, int repeats, std::optional<int> axis, StreamOrDevice s) {
        if (axis.has_value()) {
          return repeat(array, repeats, axis.value(), s);
        } else {
          return repeat(array, repeats, s);
        }
      },
      nb::arg(),
      "repeats"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def repeat(array: array, repeats: int, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("clip", [](const array& a, const std::optional<ScalarOrArray>& min, const std::optional<ScalarOrArray>& max, StreamOrDevice s) {
        std::optional<array> min_ = std::nullopt;
        std::optional<array> max_ = std::nullopt;
        if (min) {
          min_ = to_arrays(a, min.value()).second;
        }
        if (max) {
          max_ = to_arrays(a, max.value()).second;
        }
        return clip(a, min_, max_, s);
      },
      nb::arg(),
      "a_min"_a.none(),
      "a_max"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def clip(a: array, /, a_min: Union[scalar, array, None], a_max: Union[scalar, array, None], *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("pad",
      [](const array& a,
         const std::variant<
             int, std::tuple<int>, std::pair<int, int>, std::vector<std::pair<int, int>>>& pad_width, const std::string mode, const ScalarOrArray& constant_value, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&pad_width); pv) {
          return pad(a, *pv, to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::tuple<int>>(&pad_width); pv) {
          return pad(a, std::get<0>(*pv), to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::pair<int, int>>(&pad_width); pv) {
          return pad(a, *pv, to_array(constant_value), mode, s);
        } else {
          auto v = std::get<std::vector<std::pair<int, int>>>(pad_width);
          if (v.size() == 1) {
            return pad(a, v[0], to_array(constant_value), mode, s);
          } else {
            return pad(a, v, to_array(constant_value), mode, s);
          }
        }
      },
      nb::arg(),
      "pad_width"_a,
      "mode"_a = "constant",
      "constant_values"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def pad(a: array, pad_width: Union[int, tuple[int], tuple[int, int], list[tuple[int, int]]], mode: Literal['constant', 'edge'] = 'constant', constant_values: Union[scalar, array] = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("as_strided", [](const array& a, std::optional<std::vector<int>> shape, std::optional<std::vector<size_t>> strides, size_t offset, StreamOrDevice s) {
        std::vector<int> a_shape = (shape) ? *shape : a.shape();
        std::vector<size_t> a_strides;
        if (strides) {
          a_strides = *strides;
        } else {
          a_strides = std::vector<size_t>(a_shape.size(), 1);
          for (int i = a_shape.size() - 1; i > 0; i--) {
            a_strides[i - 1] = a_shape[i] * a_strides[i];
          }
        }
        return as_strided(a, a_shape, a_strides, offset, s);
      },
      nb::arg(),
      "shape"_a = nb::none(),
      "strides"_a = nb::none(),
      "offset"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def as_strided(a: array, /, shape: Optional[Sequence[int]] = None, strides: Optional[Sequence[int]] = None, offset: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cumsum", [](const array& a, std::optional<int> axis, bool reverse, bool inclusive, StreamOrDevice s) {
        if (axis) {
          return cumsum(a, *axis, reverse, inclusive, s);
        } else {
          return cumsum(reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig("def cumsum(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cumprod", [](const array& a, std::optional<int> axis, bool reverse, bool inclusive, StreamOrDevice s) {
        if (axis) {
          return cumprod(a, *axis, reverse, inclusive, s);
        } else {
          return cumprod(reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig("def cumprod(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cummax", [](const array& a, std::optional<int> axis, bool reverse, bool inclusive, StreamOrDevice s) {
        if (axis) {
          return cummax(a, *axis, reverse, inclusive, s);
        } else {
          return cummax(reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig("def cummax(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("cummin", [](const array& a, std::optional<int> axis, bool reverse, bool inclusive, StreamOrDevice s) {
        if (axis) {
          return cummin(a, *axis, reverse, inclusive, s);
        } else {
          return cummin(reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig("def cummin(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("where", [](const ScalarOrArray& condition, const ScalarOrArray& x_, const ScalarOrArray& y_, StreamOrDevice s) {
        auto [x, y] = to_arrays(x_, y_);
        return where(to_array(condition), x, y, s);
      },
      "condition"_a,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def where(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("nan_to_num", [](const ScalarOrArray& a, float nan, std::optional<float>& posinf, std::optional<float>& neginf, StreamOrDevice s) {
        return nan_to_num(to_array(a), nan, posinf, neginf, s);
      },
      nb::arg(),
      "nan"_a = 0.0f,
      "posinf"_a = nb::none(),
      "neginf"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def nan_to_num(a: Union[scalar, array], nan: float = 0, posinf: Optional[float] = None, neginf: Optional[float] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("round", [](const ScalarOrArray& a, int decimals, StreamOrDevice s) {
        return round(to_array(a), decimals, s);
      },
      nb::arg(),
      "decimals"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def round(a: array, /, decimals: int = 0, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("tile", [](const array& a, const std::variant<int, std::vector<int>>& reps, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&reps); pv) {
          return tile(a, {*pv}, s);
        } else {
          return tile(a, std::get<std::vector<int>>(reps), s);
        }
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def tile(a: array, reps: Union[int, Sequence[int]], /, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("diagonal",
      &diagonal,
      "a"_a,
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "stream"_a = nb::none(),
      nb::sig("def diagonal(a: array, offset: int = 0, axis1: int = 0, axis2: int = 1, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("diag",
      &diag,
      nb::arg(),
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def diag(a: array, /, k: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("trace", [](const array& a, int offset, int axis1, int axis2, std::optional<Dtype> dtype, StreamOrDevice s) {
        if (!dtype.has_value()) {
          return trace(a, offset, axis1, axis2, s);
        }
        return trace(a, offset, axis1, axis2, dtype.value(), s);
      },
      nb::arg(),
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def trace(a: array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def("atleast_1d",
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_1d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_1d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig("def atleast_1d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"));
  m.def("atleast_2d",
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_2d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_2d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig("def atleast_2d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"));
  m.def("atleast_3d",
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_3d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_3d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig("def atleast_3d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"));
  m.def("issubdtype", [](const nb::object& d1, const nb::object& d2) {
        auto dispatch_second = [](const auto& t1, const auto& d2) {
          if (nb::isinstance<Dtype>(d2)) {
            return issubdtype(t1, nb::cast<Dtype>(d2));
          } else if (nb::isinstance<Dtype::Category>(d2)) {
            return issubdtype(t1, nb::cast<Dtype::Category>(d2));
          } else {
            throw std::invalid_argument("[issubdtype] Received invalid type for second input.");
          }
        };
        if (nb::isinstance<Dtype>(d1)) {
          return dispatch_second(nb::cast<Dtype>(d1), d2);
        } else if (nb::isinstance<Dtype::Category>(d1)) {
          return dispatch_second(nb::cast<Dtype::Category>(d1), d2);
        } else {
          throw std::invalid_argument("[issubdtype] Received invalid type for first input.");
        }
      },
      ""_a,
      ""_a,
      nb::sig("def issubdtype(arg1: Union[Dtype, DtypeCategory], arg2: Union[Dtype, DtypeCategory]) -> bool"));
  m.def("bitwise_and", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return bitwise_and(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def bitwise_and(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("bitwise_or", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return bitwise_or(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def bitwise_or(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("bitwise_xor", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return bitwise_xor(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def bitwise_xor(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("left_shift", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return left_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def left_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("right_shift", [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return right_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def right_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"));
  m.def("view", [](const ScalarOrArray& a, const Dtype& dtype, StreamOrDevice s) {
        return view(to_array(a), dtype, s);
      },
      nb::arg(),
      "dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def view(a: Union[scalar, array], dtype: Dtype, stream: Union[None, Stream, Device] = None) -> array"));
}
