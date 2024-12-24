#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {

array arange(
    double start,
    double stop,
    double step,
    Dtype dtype,
    StreamOrDevice s = {});
array arange(double start, double stop, double step, StreamOrDevice s = {});
array arange(double start, double stop, Dtype dtype, StreamOrDevice s = {});
array arange(double start, double stop, StreamOrDevice s = {});
array arange(double stop, Dtype dtype, StreamOrDevice s = {});
array arange(double stop, StreamOrDevice s = {});

array arange(int start, int stop, int step, StreamOrDevice s = {});
array arange(int start, int stop, StreamOrDevice s = {});
array arange(int stop, StreamOrDevice s = {});

array linspace(
    double start,
    double stop,
    int num = 50,
    Dtype dtype = float32,
    StreamOrDevice s = {});

array astype(array a, Dtype dtype, StreamOrDevice s = {});

array as_strided(
    array a,
    std::vector<int> shape,
    std::vector<size_t> strides,
    size_t offset,
    StreamOrDevice s = {});

array copy(array a, StreamOrDevice s = {});

array full(
    std::vector<int> shape,
    array vals,
    Dtype dtype,
    StreamOrDevice s = {});
array full(std::vector<int> shape, array vals, StreamOrDevice s = {});
template <typename T>
array full(std::vector<int> shape, T val, Dtype dtype, StreamOrDevice s = {}) {
  return full(std::move(shape), array(val, dtype), to_stream(s));
}
template <typename T>
array full(std::vector<int> shape, T val, StreamOrDevice s = {}) {
  return full(std::move(shape), array(val), to_stream(s));
}

array zeros(const std::vector<int>& shape, Dtype dtype, StreamOrDevice s = {});
inline array zeros(const std::vector<int>& shape, StreamOrDevice s = {}) {
  return zeros(shape, float32, s);
}
array zeros_like(const array& a, StreamOrDevice s = {});

array ones(const std::vector<int>& shape, Dtype dtype, StreamOrDevice s = {});
inline array ones(const std::vector<int>& shape, StreamOrDevice s = {}) {
  return ones(shape, float32, s);
}
array ones_like(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape (n,m) with ones in the specified diagonal
 * k, and zeros everywhere else. */
array eye(int n, int m, int k, Dtype dtype, StreamOrDevice s = {});
inline array eye(int n, Dtype dtype, StreamOrDevice s = {}) {
  return eye(n, n, 0, dtype, s);
}
inline array eye(int n, int m, StreamOrDevice s = {}) {
  return eye(n, m, 0, float32, s);
}
inline array eye(int n, int m, int k, StreamOrDevice s = {}) {
  return eye(n, m, k, float32, s);
}
inline array eye(int n, StreamOrDevice s = {}) {
  return eye(n, n, 0, float32, s);
}

/** Create a square matrix of shape (n,n) of zeros, and ones in the major
 * diagonal. */
array identity(int n, Dtype dtype, StreamOrDevice s = {});
inline array identity(int n, StreamOrDevice s = {}) {
  return identity(n, float32, s);
}

array tri(int n, int m, int k, Dtype type, StreamOrDevice s = {});
inline array tri(int n, Dtype type, StreamOrDevice s = {}) {
  return tri(n, n, 0, type, s);
}

array tril(array x, int k = 0, StreamOrDevice s = {});
array triu(array x, int k = 0, StreamOrDevice s = {});

array reshape(const array& a, std::vector<int> shape, StreamOrDevice s = {});

array flatten(
    const array& a,
    int start_axis,
    int end_axis = -1,
    StreamOrDevice s = {});

array flatten(const array& a, StreamOrDevice s = {});

array squeeze(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

array squeeze(const array& a, int axis, StreamOrDevice s = {});

array squeeze(const array& a, StreamOrDevice s = {});

array expand_dims(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

array expand_dims(const array& a, int axis, StreamOrDevice s = {});

array slice(
    const array& a,
    std::vector<int> start,
    std::vector<int> stop,
    std::vector<int> strides,
    StreamOrDevice s = {});

array slice(
    const array& a,
    std::vector<int> start,
    std::vector<int> stop,
    StreamOrDevice s = {});

array slice_update(
    const array& src,
    const array& update,
    std::vector<int> start,
    std::vector<int> stop,
    std::vector<int> strides,
    StreamOrDevice s = {});

array slice_update(
    const array& src,
    const array& update,
    std::vector<int> start,
    std::vector<int> stop,
    StreamOrDevice s = {});

std::vector<array>
split(const array& a, int num_splits, int axis, StreamOrDevice s = {});
std::vector<array> split(const array& a, int num_splits, StreamOrDevice s = {});
std::vector<array> split(
    const array& a,
    const std::vector<int>& indices,
    int axis,
    StreamOrDevice s = {});
std::vector<array>
split(const array& a, const std::vector<int>& indices, StreamOrDevice s = {});

std::vector<array> meshgrid(
    const std::vector<array>& arrays,
    bool sparse = false,
    std::string indexing = "xy",
    StreamOrDevice s = {});

array clip(
    const array& a,
    const std::optional<array>& a_min = std::nullopt,
    const std::optional<array>& a_max = std::nullopt,
    StreamOrDevice s = {});

array concatenate(
    const std::vector<array>& arrays,
    int axis,
    StreamOrDevice s = {});
array concatenate(const std::vector<array>& arrays, StreamOrDevice s = {});

array stack(const std::vector<array>& arrays, int axis, StreamOrDevice s = {});
array stack(const std::vector<array>& arrays, StreamOrDevice s = {});

array repeat(const array& arr, int repeats, int axis, StreamOrDevice s = {});
array repeat(const array& arr, int repeats, StreamOrDevice s = {});

array tile(const array& arr, std::vector<int> reps, StreamOrDevice s = {});

array transpose(const array& a, std::vector<int> axes, StreamOrDevice s = {});
inline array transpose(
    const array& a,
    std::initializer_list<int> axes,
    StreamOrDevice s = {}) {
  return transpose(a, std::vector<int>(axes), s);
}

array swapaxes(const array& a, int axis1, int axis2, StreamOrDevice s = {});

array moveaxis(
    const array& a,
    int source,
    int destination,
    StreamOrDevice s = {});

array pad(
    const array& a,
    const std::vector<int>& axes,
    const std::vector<int>& low_pad_size,
    const std::vector<int>& high_pad_size,
    const array& pad_value = array(0),
    const std::string mode = "constant",
    StreamOrDevice s = {});

array pad(
    const array& a,
    const std::vector<std::pair<int, int>>& pad_width,
    const array& pad_value = array(0),
    const std::string mode = "constant",
    StreamOrDevice s = {});
array pad(
    const array& a,
    const std::pair<int, int>& pad_width,
    const array& pad_value = array(0),
    const std::string mode = "constant",
    StreamOrDevice s = {});
array pad(
    const array& a,
    int pad_width,
    const array& pad_value = array(0),
    const std::string mode = "constant",
    StreamOrDevice s = {});

array transpose(const array& a, StreamOrDevice s = {});

array broadcast_to(
    const array& a,
    const std::vector<int>& shape,
    StreamOrDevice s = {});

std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    StreamOrDevice s = {});

array equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator==(const array& a, const array& b) {
  return equal(a, b);
}
template <typename T>
array operator==(T a, const array& b) {
  return equal(array(a), b);
}
template <typename T>
array operator==(const array& a, T b) {
  return equal(a, array(b));
}

array not_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator!=(const array& a, const array& b) {
  return not_equal(a, b);
}
template <typename T>
array operator!=(T a, const array& b) {
  return not_equal(array(a), b);
}
template <typename T>
array operator!=(const array& a, T b) {
  return not_equal(a, array(b));
}

array greater(const array& a, const array& b, StreamOrDevice s = {});
inline array operator>(const array& a, const array& b) {
  return greater(a, b);
}
template <typename T>
array operator>(T a, const array& b) {
  return greater(array(a), b);
}
template <typename T>
array operator>(const array& a, T b) {
  return greater(a, array(b));
}

array greater_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator>=(const array& a, const array& b) {
  return greater_equal(a, b);
}
template <typename T>
array operator>=(T a, const array& b) {
  return greater_equal(array(a), b);
}
template <typename T>
array operator>=(const array& a, T b) {
  return greater_equal(a, array(b));
}

array less(const array& a, const array& b, StreamOrDevice s = {});
inline array operator<(const array& a, const array& b) {
  return less(a, b);
}
template <typename T>
array operator<(T a, const array& b) {
  return less(array(a), b);
}
template <typename T>
array operator<(const array& a, T b) {
  return less(a, array(b));
}

array less_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator<=(const array& a, const array& b) {
  return less_equal(a, b);
}
template <typename T>
array operator<=(T a, const array& b) {
  return less_equal(array(a), b);
}
template <typename T>
array operator<=(const array& a, T b) {
  return less_equal(a, array(b));
}

array array_equal(
    const array& a,
    const array& b,
    bool equal_nan,
    StreamOrDevice s = {});
inline array
array_equal(const array& a, const array& b, StreamOrDevice s = {}) {
  return array_equal(a, b, false, s);
}

array isnan(const array& a, StreamOrDevice s = {});

array isinf(const array& a, StreamOrDevice s = {});

array isfinite(const array& a, StreamOrDevice s = {});

array isposinf(const array& a, StreamOrDevice s = {});

array isneginf(const array& a, StreamOrDevice s = {});

array where(
    const array& condition,
    const array& x,
    const array& y,
    StreamOrDevice s = {});

array nan_to_num(
    const array& a,
    float nan = 0.0f,
    const std::optional<float> posinf = std::nullopt,
    const std::optional<float> neginf = std::nullopt,
    StreamOrDevice s = {});

array all(const array& a, bool keepdims, StreamOrDevice s = {});
inline array all(const array& a, StreamOrDevice s = {}) {
  return all(a, false, to_stream(s));
}

array allclose(
    const array& a,
    const array& b,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false,
    StreamOrDevice s = {});

/** Returns a boolean array where two arrays are element-wise equal within the
 * specified tolerance. */
array isclose(
    const array& a,
    const array& b,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false,
    StreamOrDevice s = {});

array all(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array all(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array any(const array& a, bool keepdims, StreamOrDevice s = {});
inline array any(const array& a, StreamOrDevice s = {}) {
  return any(a, false, to_stream(s));
}

array any(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array any(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array sum(const array& a, bool keepdims, StreamOrDevice s = {});
inline array sum(const array& a, StreamOrDevice s = {}) {
  return sum(a, false, to_stream(s));
}

array sum(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array sum(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array mean(const array& a, bool keepdims, StreamOrDevice s = {});
inline array mean(const array& a, StreamOrDevice s = {}) {
  return mean(a, false, to_stream(s));
}

array mean(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array mean(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array var(const array& a, bool keepdims, int ddof = 0, StreamOrDevice s = {});
inline array var(const array& a, StreamOrDevice s = {}) {
  return var(a, false, 0, to_stream(s));
}

/** Computes the variance of the elements of an array along the given
 * axes */
array var(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the variance of the elements of an array along the given
 * axis */
array var(
    const array& a,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

array std(const array& a, bool keepdims, int ddof = 0, StreamOrDevice s = {});
inline array std(const array& a, StreamOrDevice s = {}) {
  return std(a, false, 0, to_stream(s));
}

/** Computes the standard deviatoin of the elements of an array along the given
 * axes */
array std(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the standard deviation of the elements of an array along the given
 * axis */
array std(
    const array& a,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

array prod(const array& a, bool keepdims, StreamOrDevice s = {});
inline array prod(const array& a, StreamOrDevice s = {}) {
  return prod(a, false, to_stream(s));
}

array prod(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array prod(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array max(const array& a, bool keepdims, StreamOrDevice s = {});
inline array max(const array& a, StreamOrDevice s = {}) {
  return max(a, false, to_stream(s));
}

array max(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array max(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array min(const array& a, bool keepdims, StreamOrDevice s = {});
inline array min(const array& a, StreamOrDevice s = {}) {
  return min(a, false, to_stream(s));
}

array min(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array min(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array argmin(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmin(const array& a, StreamOrDevice s = {}) {
  return argmin(a, false, s);
}

array argmin(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array argmax(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmax(const array& a, StreamOrDevice s = {}) {
  return argmax(a, false, s);
}

array argmax(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array logsumexp(const array& a, bool keepdims, StreamOrDevice s = {});
inline array logsumexp(const array& a, StreamOrDevice s = {}) {
  return logsumexp(a, false, to_stream(s));
}

array logsumexp(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

array logsumexp(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

array abs(const array& a, StreamOrDevice s = {});

array negative(const array& a, StreamOrDevice s = {});
array operator-(const array& a);

array sign(const array& a, StreamOrDevice s = {});

array logical_not(const array& a, StreamOrDevice s = {});

array logical_and(const array& a, const array& b, StreamOrDevice s = {});
array operator&&(const array& a, const array& b);

array logical_or(const array& a, const array& b, StreamOrDevice s = {});
array operator||(const array& a, const array& b);

array reciprocal(const array& a, StreamOrDevice s = {});

array add(const array& a, const array& b, StreamOrDevice s = {});
array operator+(const array& a, const array& b);
template <typename T>
array operator+(T a, const array& b) {
  return add(array(a), b);
}
template <typename T>
array operator+(const array& a, T b) {
  return add(a, array(b));
}

array subtract(const array& a, const array& b, StreamOrDevice s = {});
array operator-(const array& a, const array& b);
template <typename T>
array operator-(T a, const array& b) {
  return subtract(array(a), b);
}
template <typename T>
array operator-(const array& a, T b) {
  return subtract(a, array(b));
}

array multiply(const array& a, const array& b, StreamOrDevice s = {});
array operator*(const array& a, const array& b);
template <typename T>
array operator*(T a, const array& b) {
  return multiply(array(a), b);
}
template <typename T>
array operator*(const array& a, T b) {
  return multiply(a, array(b));
}

array divide(const array& a, const array& b, StreamOrDevice s = {});
array operator/(const array& a, const array& b);
array operator/(double a, const array& b);
array operator/(const array& a, double b);

std::vector<array>
divmod(const array& a, const array& b, StreamOrDevice s = {});

array floor_divide(const array& a, const array& b, StreamOrDevice s = {});

array remainder(const array& a, const array& b, StreamOrDevice s = {});
array operator%(const array& a, const array& b);
template <typename T>
array operator%(T a, const array& b) {
  return remainder(array(a), b);
}
template <typename T>
array operator%(const array& a, T b) {
  return remainder(a, array(b));
}

array maximum(const array& a, const array& b, StreamOrDevice s = {});

array minimum(const array& a, const array& b, StreamOrDevice s = {});

array floor(const array& a, StreamOrDevice s = {});

array ceil(const array& a, StreamOrDevice s = {});

array square(const array& a, StreamOrDevice s = {});

array exp(const array& a, StreamOrDevice s = {});

array sin(const array& a, StreamOrDevice s = {});

array cos(const array& a, StreamOrDevice s = {});

array tan(const array& a, StreamOrDevice s = {});

array sinh(const array& a, StreamOrDevice s = {});

array cosh(const array& a, StreamOrDevice s = {});

array tanh(const array& a, StreamOrDevice s = {});

array degrees(const array& a, StreamOrDevice s = {});

array radians(const array& a, StreamOrDevice s = {});

array log(const array& a, StreamOrDevice s = {});

array log2(const array& a, StreamOrDevice s = {});

array log10(const array& a, StreamOrDevice s = {});

array log1p(const array& a, StreamOrDevice s = {});

array sigmoid(const array& a, StreamOrDevice s = {});

array erf(const array& a, StreamOrDevice s = {});

array erfinv(const array& a, StreamOrDevice s = {});

array expm1(const array& a, StreamOrDevice s = {});

array stop_gradient(const array& a, StreamOrDevice s = {});

array round(const array& a, int decimals, StreamOrDevice s = {});
inline array round(const array& a, StreamOrDevice s = {}) {
  return round(a, 0, s);
}

array matmul(const array& a, const array& b, StreamOrDevice s = {});

array gather(
    const array& a,
    const std::vector<array>& indices,
    const std::vector<int>& axes,
    const std::vector<int>& slice_sizes,
    StreamOrDevice s = {});
inline array gather(
    const array& a,
    const array& indices,
    int axis,
    const std::vector<int>& slice_sizes,
    StreamOrDevice s = {}) {
  return gather(a, {indices}, std::vector<int>{axis}, slice_sizes, s);
}

array take(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});
array take(const array& a, int index, int axis, StreamOrDevice s = {});

array take(const array& a, const array& indices, StreamOrDevice s = {});
array take(const array& a, int index, StreamOrDevice s = {});

array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});

array put_along_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    StreamOrDevice s = {});

/** Scatter updates to the given indices.
 *
 * The parameters ``indices`` and ``axes`` determine the locations of ``a``
 * that are updated with the values in ``updates``. Assuming 1-d ``indices``
 * for simplicity, ``indices[i]`` are the indices on axis ``axes[i]`` to which
 * the values in ``updates`` will be applied. Note each array in
 * ``indices`` is assigned to a corresponding axis and hence ``indices.size() ==
 * axes.size()``. If an index/axis pair is not provided then indices along that
 * axis are assumed to be zero.
 *
 * Note the rank of ``updates`` must be equal to the sum of the rank of the
 * broadcasted ``indices`` and the rank of ``a``. In other words, assuming the
 * arrays in ``indices`` have the same shape, ``updates.ndim() ==
 * indices[0].ndim() + a.ndim()``. The leading dimensions of ``updates``
 * correspond to the indices, and the remaining ``a.ndim()`` dimensions are the
 * values that will be applied to the given location in ``a``.
 *
 * For example:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = array({2});
 * auto updates = reshape(arange(1, 3, float32), {1, 1, 2});
 * std::vector<int> axes{0};
 *
 * auto out = scatter(in, {indices}, updates, axes);
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [1, 2, 0, 0],
 *        [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * This scatters the two-element row vector ``[1, 2]`` starting at the ``(2,
 * 0)`` position of ``a``.
 *
 * Adding another element to ``indices`` will scatter into another location of
 * ``a``. We also have to add an another update for the new index:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = array({2, 0});
 * auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
 * std::vector<int> axes{0};
 *
 * auto out = scatter(in, {indices}, updates, axes):
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[3, 4, 0, 0],
 *        [0, 0, 0, 0],
 *        [1, 2, 0, 0],
 *        [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * To control the scatter location on an additional axis, add another index
 * array to ``indices`` and another axis to ``axes``:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = std::vector{array({2, 0}), array({1, 2})};
 * auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
 * std::vector<int> axes{0, 1};
 *
 * auto out = scatter(in, indices, updates, axes);
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[0, 0, 3, 4],
 *       [0, 0, 0, 0],
 *       [0, 1, 2, 0],
 *       [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * Items in indices are broadcasted together. This means:
 *
 * @code
 * auto indices = std::vector{array({2, 0}), array({1})};
 * @endcode
 *
 * is equivalent to:
 *
 * @code
 * auto indices = std::vector{array({2, 0}), array({1, 1})};
 * @endcode
 *
 * Note, ``scatter`` does not perform bounds checking on the indices and
 * updates.  Out-of-bounds accesses on ``a`` are undefined and typically result
 * in unintended or invalid memory writes.
 */
array scatter(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter(a, {indices}, updates, std::vector<int>{axis}, s);
}

array scatter_add(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_add(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_add(a, {indices}, updates, std::vector<int>{axis}, s);
}

array scatter_prod(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_prod(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_prod(a, {indices}, updates, std::vector<int>{axis}, s);
}

array scatter_max(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_max(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_max(a, {indices}, updates, std::vector<int>{axis}, s);
}
array scatter_min(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_min(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_min(a, {indices}, updates, std::vector<int>{axis}, s);
}

array sqrt(const array& a, StreamOrDevice s = {});

array rsqrt(const array& a, StreamOrDevice s = {});

array softmax(
    const array& a,
    const std::vector<int>& axes,
    bool precise = false,
    StreamOrDevice s = {});

array softmax(const array& a, bool precise = false, StreamOrDevice s = {});

inline array
softmax(const array& a, int axis, bool precise = false, StreamOrDevice s = {}) {
  return softmax(a, std::vector<int>{axis}, precise, s);
}

array power(const array& a, const array& b, StreamOrDevice s = {});

array cumsum(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

array cumprod(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

array cummax(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

array cummin(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

array diagonal(
    const array& a,
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    StreamOrDevice s = {});

array diag(const array& a, int k = 0, StreamOrDevice s = {});

array trace(
    const array& a,
    int offset,
    int axis1,
    int axis2,
    Dtype dtype,
    StreamOrDevice s = {});
array trace(
    const array& a,
    int offset,
    int axis1,
    int axis2,
    StreamOrDevice s = {});
array trace(const array& a, StreamOrDevice s = {});

std::vector<array> depends(
    const std::vector<array>& inputs,
    const std::vector<array>& dependencies);

array atleast_1d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_1d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
array atleast_2d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_2d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
array atleast_3d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_3d(
    const std::vector<array>& a,
    StreamOrDevice s = {});

array number_of_elements(
    const array& a,
    std::vector<int> axes,
    bool inverted,
    Dtype dtype = int32,
    StreamOrDevice s = {});

array bitwise_and(const array& a, const array& b, StreamOrDevice s = {});
array operator&(const array& a, const array& b);

array bitwise_or(const array& a, const array& b, StreamOrDevice s = {});
array operator|(const array& a, const array& b);

array bitwise_xor(const array& a, const array& b, StreamOrDevice s = {});
array operator^(const array& a, const array& b);

array left_shift(const array& a, const array& b, StreamOrDevice s = {});
array operator<<(const array& a, const array& b);

array right_shift(const array& a, const array& b, StreamOrDevice s = {});
array operator>>(const array& a, const array& b);

array view(const array& a, const Dtype& dtype, StreamOrDevice s = {});

}
