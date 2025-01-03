#include <numeric>
#include <sstream>

#include "python/src/convert.h"
#include "python/src/indexing.h"

#include "mlx/ops.h"

bool is_none_slice(const nb::slice& in_slice) {
  return (
      nb::getattr(in_slice, "start").is_none() &&
      nb::getattr(in_slice, "stop").is_none() &&
      nb::getattr(in_slice, "step").is_none());
}

int get_slice_int(nb::object obj, int default_val) {
  if (!obj.is_none()) {
    if (!nb::isinstance<nb::int_>(obj)) {
      throw std::invalid_argument("Slice indices must be integers or None.");
    }
    return nb::cast<int>(nb::cast<nb::int_>(obj));
  }
  return default_val;
}

void get_slice_params(
    int& starts, int& ends, int& strides, const nb::slice& in_slice, int axis_size) {

  strides = get_slice_int(nb::getattr(in_slice, "step"), 1);
  starts = get_slice_int(
      nb::getattr(in_slice, "start"), strides < 0 ? axis_size - 1 : 0);
  ends = get_slice_int(
      nb::getattr(in_slice, "stop"), strides < 0 ? -axis_size - 1 : axis_size);
}

array get_int_index(nb::object idx, int axis_size) {
  int idx_ = nb::cast<int>(idx);
  idx_ = (idx_ < 0) ? idx_ + axis_size : idx_;

  return array(idx_, uint32);
}

bool is_valid_index_type(const nb::object& obj) {
  return nb::isinstance<nb::slice>(obj) || nb::isinstance<nb::int_>(obj) ||
      nb::isinstance<array>(obj) || obj.is_none() || nb::ellipsis().is(obj) ||
      nb::isinstance<nb::list>(obj);
}

array mlx_get_item_slice(const array& src, const nb::slice& in_slice) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  if (is_none_slice(in_slice)) {
    return src;
  }

  std::vector<int> starts(src.ndim(), 0);
  std::vector<int> ends = src.shape();
  std::vector<int> strides(src.ndim(), 1);

  get_slice_params(starts[0], ends[0], strides[0], in_slice, ends[0]);
  return slice(src, starts, ends, strides);
}

array mlx_get_item_array(const array& src, const array& indices) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  if (indices.dtype() == bool_) {
    throw std::invalid_argument("boolean indices are not yet supported");
  }

  return take(src, indices, 0);
}

array mlx_get_item_int(const array& src, const nb::int_& idx) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  return take(src, get_int_index(idx, src.shape(0)), 0);
}

array mlx_gather_nd(
    array src, const std::vector<nb::object>& indices, bool gather_first, int& max_dims) {
  max_dims = 0;
  std::vector<array> gather_indices;
  std::vector<bool> is_slice(indices.size(), false);
  int num_slices = 0;
  for (int i = 0; i < indices.size(); i++) {
    auto& idx = indices[i];

    if (nb::isinstance<nb::slice>(idx)) {
      int start, end, stride;
      get_slice_params(
          start, end, stride, nb::cast<nb::slice>(idx), src.shape(i));

      start = (start < 0) ? start + src.shape(i) : start;
      end = (end < 0) ? end + src.shape(i) : end;

      gather_indices.push_back(arange(start, end, stride, uint32));
      num_slices++;
      is_slice[i] = true;
    } else if (nb::isinstance<nb::int_>(idx)) {
      gather_indices.push_back(get_int_index(idx, src.shape(i)));
    } else if (nb::isinstance<array>(idx)) {
      auto arr = nb::cast<array>(idx);
      max_dims = std::max(static_cast<int>(arr.ndim()), max_dims);
      gather_indices.push_back(arr);
    }
  }

  if (gather_first) {
    int slice_index = 0;
    for (int i = 0; i < gather_indices.size(); i++) {
      if (is_slice[i]) {
        std::vector<int> index_shape(max_dims + num_slices, 1);
        index_shape[max_dims + slice_index] = gather_indices[i].shape(0);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
        slice_index++;
      } else {
        std::vector<int> index_shape = gather_indices[i].shape();
        index_shape.insert(index_shape.end(), num_slices, 1);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
      }
    }
  } else {
    for (int i = 0; i < gather_indices.size(); i++) {
      if (i < num_slices) {
        std::vector<int> index_shape(max_dims + num_slices, 1);
        index_shape[i] = gather_indices[i].shape(0);
        gather_indices[i] = reshape(gather_indices[i], index_shape);
      }
    }
  }

  std::vector<int> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<int> slice_sizes = src.shape();
  std::fill(slice_sizes.begin(), slice_sizes.begin() + indices.size(), 1);
  src = gather(src, gather_indices, axes, slice_sizes);

  std::vector<int> out_shape;
  out_shape.insert(
      out_shape.end(),
      src.shape().begin(),
      src.shape().begin() + max_dims + num_slices);
  out_shape.insert(
      out_shape.end(),
      src.shape().begin() + max_dims + num_slices + indices.size(),
      src.shape().end());
  src = reshape(src, out_shape);

  return src;
}

auto mlx_expand_ellipsis(
    const std::vector<int>& shape, const nb::tuple& entries) {
  std::vector<nb::object> indices;

  int non_none_indices_before = 0;
  int non_none_indices_after = 0;
  std::vector<nb::object> r_indices;
  int i = 0;
  bool has_ellipsis = false;

  for (; i < entries.size(); i++) {
    auto idx = entries[i];
    if (!is_valid_index_type(idx)) {
      throw std::invalid_argument("Cannot index mlx array using the given type yet");
    }
    if (!nb::ellipsis().is(idx)) {
      indices.push_back(idx);
      non_none_indices_before += !idx.is_none();
    } else {
      has_ellipsis = true;
      break;
    }
  }

  for (int j = entries.size() - 1; j > i; j--) {
    auto idx = entries[j];
    if (!is_valid_index_type(idx)) {
      throw std::invalid_argument("Cannot index mlx array using the given type yet");
    }
    if (nb::ellipsis().is(idx)) {
      throw std::invalid_argument("An index can only have a single ellipsis (...)");
    }
    r_indices.push_back(idx);
    non_none_indices_after += !idx.is_none();
  }

  int non_none_indices = non_none_indices_before + non_none_indices_after;

  if (has_ellipsis) {
    for (int axis = non_none_indices_before;
         axis < shape.size() - non_none_indices_after;
         axis++) {
      indices.push_back(nb::slice(0, shape[axis], 1));
      non_none_indices++;
    }
  }

  indices.insert(indices.end(), r_indices.rbegin(), r_indices.rend());

  return std::make_pair(non_none_indices, indices);
}

array mlx_get_item_nd(array src, const nb::tuple& entries) {
  if (entries.size() == 0) {
    return src;
  }

  auto [non_none_indices, indices] = mlx_expand_ellipsis(src.shape(), entries);
  for (auto& idx : indices) {
    if (nb::isinstance<nb::list>(idx)) {
      idx = nb::cast(array_from_list(nb::cast<nb::list>(idx), {}));
    }
  }

  if (non_none_indices > src.ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << src.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  std::vector<nb::object> remaining_indices;
  bool have_array = false;
  {
    bool have_non_array = false;
    bool gather_first = false;
    for (auto& idx : indices) {
      if (nb::isinstance<array>(idx) || (nb::isinstance<nb::int_>(idx))) {
        if (have_array && have_non_array) {
          gather_first = true;
          break;
        }
        have_array = true;
      } else {
        have_non_array |= have_array;
      }
    }

    int n_arr = 0;
    for (auto& idx : indices) {
      n_arr += nb::isinstance<array>(idx);
    }

    have_array &= n_arr > 0;

    if (have_array) {
      int last_array;
      for (last_array = indices.size() - 1; last_array >= 0; last_array--) {
        auto& idx = indices[last_array];
        if (nb::isinstance<array>(idx) || nb::isinstance<nb::int_>(idx)) {
          break;
        }
      }

      std::vector<nb::object> gather_indices;
      for (int i = 0; i <= last_array; i++) {
        auto& idx = indices[i];
        if (!idx.is_none()) {
          gather_indices.push_back(idx);
        }
      }
      int max_dims;
      src = mlx_gather_nd(src, gather_indices, gather_first, max_dims);

      if (gather_first) {
        for (int i = 0; i < max_dims; i++) {
          remaining_indices.push_back(
              nb::slice(nb::none(), nb::none(), nb::none()));
        }
        for (int i = 0; i < last_array; i++) {
          auto& idx = indices[i];
          if (idx.is_none()) {
            remaining_indices.push_back(indices[i]);
          } else if (nb::isinstance<nb::slice>(idx)) {
            remaining_indices.push_back(
                nb::slice(nb::none(), nb::none(), nb::none()));
          }
        }
        for (int i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      } else {
        for (int i = 0; i < indices.size(); i++) {
          auto& idx = indices[i];
          if (nb::isinstance<array>(idx) || nb::isinstance<nb::int_>(idx)) {
            break;
          } else if (idx.is_none()) {
            remaining_indices.push_back(idx);
          } else {
            remaining_indices.push_back(
                nb::slice(nb::none(), nb::none(), nb::none()));
          }
        }
        for (int i = 0; i < max_dims; i++) {
          remaining_indices.push_back(
              nb::slice(nb::none(), nb::none(), nb::none()));
        }
        for (int i = last_array + 1; i < indices.size(); i++) {
          remaining_indices.push_back(indices[i]);
        }
      }
    }
  }
  if (have_array && remaining_indices.empty()) {
    return src;
  }
  if (remaining_indices.empty()) {
    remaining_indices = indices;
  }

  bool squeeze_needed = false;
  bool unsqueeze_needed = false;

  {
    std::vector<int> starts(src.ndim(), 0);
    std::vector<int> ends = src.shape();
    std::vector<int> strides(src.ndim(), 1);
    int axis = 0;
    for (auto& idx : remaining_indices) {
      if (!idx.is_none()) {
        if (!have_array && nb::isinstance<nb::int_>(idx)) {
          int st = nb::cast<int>(idx);
          st = (st < 0) ? st + src.shape(axis) : st;

          starts[axis] = st;
          ends[axis] = st + 1;

          squeeze_needed = true;

        } else {
          get_slice_params(
              starts[axis],
              ends[axis],
              strides[axis],
              nb::cast<nb::slice>(idx),
              ends[axis]);
        }

        axis++;
      } else {
        unsqueeze_needed = true;
      }
    }
    src = slice(src, starts, ends, strides);
  }

  if (unsqueeze_needed || squeeze_needed) {
    std::vector<int> out_shape;
    int axis = 0;
    for (auto& idx : remaining_indices) {
      if (unsqueeze_needed && idx.is_none()) {
        out_shape.push_back(1);
      } else if (squeeze_needed && nb::isinstance<nb::int_>(idx)) {
        axis++;
      } else {
        out_shape.push_back(src.shape(axis++));
      }
    }

    out_shape.insert(
        out_shape.end(), src.shape().begin() + axis, src.shape().end());

    src = reshape(src, out_shape);
  }

  return src;
}

array mlx_get_item(const array& src, const nb::object& obj) {
  if (nb::isinstance<nb::slice>(obj)) {
    return mlx_get_item_slice(src, nb::cast<nb::slice>(obj));
  } else if (nb::isinstance<array>(obj)) {
    return mlx_get_item_array(src, nb::cast<array>(obj));
  } else if (nb::isinstance<nb::int_>(obj)) {
    return mlx_get_item_int(src, nb::cast<nb::int_>(obj));
  } else if (nb::isinstance<nb::tuple>(obj)) {
    return mlx_get_item_nd(src, nb::cast<nb::tuple>(obj));
  } else if (nb::isinstance<nb::ellipsis>(obj)) {
    return src;
  } else if (obj.is_none()) {
    std::vector<int> s(1, 1);
    s.insert(s.end(), src.shape().begin(), src.shape().end());
    return reshape(src, s);
  } else if (nb::isinstance<nb::list>(obj)) {
    return mlx_get_item_array(
        src, array_from_list(nb::cast<nb::list>(obj), {}));
  }
  throw std::invalid_argument("Cannot index mlx array using the given type.");
}

std::tuple<std::vector<array>, array, std::vector<int>> mlx_scatter_args_int(
    const array& src, const nb::int_& idx, const array& update) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++)
    ;
  auto up_shape = std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto shape = src.shape();
  shape[0] = 1;

  return {
      {get_int_index(idx, src.shape(0))},
      broadcast_to(reshape(update, up_shape), shape),
      {0}};
}

std::tuple<std::vector<array>, array, std::vector<int>> mlx_scatter_args_array(
    const array& src, const array& indices, const array& update) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++)
    ;
  auto up_shape = std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto up = reshape(update, up_shape);

  up_shape = indices.shape();
  up_shape.insert(up_shape.end(), src.shape().begin() + 1, src.shape().end());
  up = broadcast_to(up, up_shape);
  up_shape.insert(up_shape.begin() + indices.ndim(), 1);
  up = reshape(up, up_shape);

  return {{indices}, up, {0}};
}

std::tuple<std::vector<array>, array, std::vector<int>> mlx_scatter_args_slice(
    const array& src, const nb::slice& in_slice, const array& update) {
  if (src.ndim() == 0) {
    throw std::invalid_argument("too many indices for array: array is 0-dimensional");
  }

  if (is_none_slice(in_slice)) {
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++)
      ;
    auto up_shape = std::vector<int>(update.shape().begin() + s, update.shape().end());
    return {{}, broadcast_to(reshape(update, up_shape), src.shape()), {}};
  }

  int start = 0;
  int end = src.shape(0);
  int stride = 1;

  get_slice_params(start, end, stride, in_slice, end);

  if (stride == 1) {
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++)
      ;
    auto up_shape = std::vector<int>(update.shape().begin() + s, update.shape().end());
    auto up = reshape(update, up_shape);

    auto idx = array({start}, {1}, uint32);

    int slice_size = (end - start);

    std::vector<int> up_shape_broadcast = {1, slice_size};
    up_shape_broadcast.insert(
        up_shape_broadcast.end(), src.shape().begin() + 1, src.shape().end());

    up = broadcast_to(up, up_shape_broadcast);

    auto indices = std::vector<array>{idx};
    auto axes = std::vector<int>{0};

    return {indices, up, axes};
  }

  return mlx_scatter_args_array(
      src, arange(start, end, stride, uint32), update);
}

std::tuple<std::vector<array>, array, std::vector<int>> mlx_scatter_args_nd(
    const array& src, const nb::tuple& entries, const array& update) {
  auto [non_none_indices, indices] = mlx_expand_ellipsis(src.shape(), entries);

  for (auto& idx : indices) {
    if (nb::isinstance<nb::list>(idx)) {
      idx = nb::cast(array_from_list(nb::cast<nb::list>(idx), {}));
    }
  }

  if (non_none_indices > src.ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << src.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  int s = 0;
  for (; s < update.ndim() && update.shape(s) == 1; s++) {
  };
  auto up_shape = std::vector<int>(update.shape().begin() + s, update.shape().end());
  auto up = reshape(update, up_shape);

  if (non_none_indices == 0) {
    return {{}, broadcast_to(up, src.shape()), {}};
  }

  unsigned long max_dim = 0;
  bool arrays_first = false;
  int num_none = 0;
  int num_slices = 0;
  int num_arrays = 0;
  int num_strided_slices = 0;
  int num_simple_slices_post = 0;
  {
    bool have_array = false;
    bool have_non_array = false;
    for (auto& idx : indices) {
      if (idx.is_none()) {
        have_non_array = have_array;
        num_none++;

      } else if (nb::isinstance<nb::slice>(idx)) {
        have_non_array = have_array;
        num_slices++;

        auto slice = nb::cast<nb::slice>(idx);
        int stride = get_slice_int(nb::getattr(slice, "step"), 1);
        if (stride != 1) {
          num_strided_slices++;
          num_simple_slices_post = 0;
        } else {
          num_simple_slices_post++;
        }

      } else if (nb::isinstance<array>(idx)) {
        have_array = true;
        if (have_array && have_non_array) {
          arrays_first = true;
        }
        max_dim = std::max(nb::cast<array>(idx).ndim(), max_dim);
        num_arrays++;
        num_simple_slices_post = 0;
      }
    }
  }

  int idx_ndim = max_dim + num_none + num_slices - num_simple_slices_post;

  idx_ndim = idx_ndim == 0 ? 1 : idx_ndim;

  std::vector<array> arr_indices;
  int slice_num = 0;
  int array_num = 0;
  int ax = 0;

  std::vector<int> update_shape(non_none_indices, 1);
  std::vector<int> slice_shapes;

  for (int i = 0; i < indices.size(); ++i) {
    auto& pyidx = indices[i];
    if (nb::isinstance<nb::slice>(pyidx)) {
      int start, end, stride;
      auto axis_size = src.shape(ax++);
      get_slice_params(
          start, end, stride, nb::cast<nb::slice>(pyidx), axis_size);

      start = (start < 0) ? start + axis_size : start;
      end = (end < 0) ? end + axis_size : end;

      std::vector<int> idx_shape(idx_ndim, 1);

      if (array_num >= num_arrays && num_strided_slices <= 0 && stride == 1) {
        auto idx = array({start}, idx_shape, uint32);
        slice_shapes.push_back(end - start);
        arr_indices.push_back(idx);

        update_shape[ax - 1] = slice_shapes.back();
      }
      else {
        auto idx = arange(start, end, stride, uint32);
        auto loc = slice_num + (arrays_first ? max_dim : 0);
        idx_shape[loc] = idx.size();
        arr_indices.push_back(reshape(idx, idx_shape));

        slice_num++;
        num_strided_slices--;

        update_shape[ax - 1] = 1;
      }
    } else if (nb::isinstance<nb::int_>(pyidx)) {
      arr_indices.push_back(get_int_index(pyidx, src.shape(ax++)));
      update_shape[ax - 1] = 1;
    } else if (pyidx.is_none()) {
      slice_num++;
    } else if (nb::isinstance<array>(pyidx)) {
      ax++;
      auto idx = nb::cast<array>(pyidx);
      std::vector<int> idx_shape(idx_ndim, 1);

      int st = (!arrays_first) * slice_num + max_dim - idx.ndim();
      for (int j = 0; j < idx.ndim(); j++) {
        idx_shape[st + j] = idx.shape()[j];
      }
      arr_indices.push_back(reshape(idx, idx_shape));
      if (!arrays_first && ++array_num == num_arrays) {
        slice_num += max_dim;
      }

      update_shape[ax - 1] = 1;
    } else {
      throw std::invalid_argument("Cannot index mlx array using the given type yet");
    }
  }

  arr_indices = broadcast_arrays(arr_indices);
  auto up_shape_broadcast = arr_indices[0].shape();

  up_shape_broadcast.insert(
      up_shape_broadcast.end(), slice_shapes.begin(), slice_shapes.end());
  up_shape_broadcast.insert(
      up_shape_broadcast.end(),
      src.shape().begin() + non_none_indices,
      src.shape().end());
  up = broadcast_to(up, up_shape_broadcast);

  auto up_reshape = arr_indices[0].shape();
  up_reshape.insert(up_reshape.end(), update_shape.begin(), update_shape.end());
  up_reshape.insert(
      up_reshape.end(),
      src.shape().begin() + non_none_indices,
      src.shape().end());

  up = reshape(up, up_reshape);

  std::vector<int> axes(arr_indices.size(), 0);
  std::iota(axes.begin(), axes.end(), 0);

  return {arr_indices, up, axes};
}

std::tuple<std::vector<array>, array, std::vector<int>>
mlx_compute_scatter_args(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto vals = to_array(v, src.dtype());
  if (nb::isinstance<nb::slice>(obj)) {
    return mlx_scatter_args_slice(src, nb::cast<nb::slice>(obj), vals);
  } else if (nb::isinstance<array>(obj)) {
    return mlx_scatter_args_array(src, nb::cast<array>(obj), vals);
  } else if (nb::isinstance<nb::int_>(obj)) {
    return mlx_scatter_args_int(src, nb::cast<nb::int_>(obj), vals);
  } else if (nb::isinstance<nb::tuple>(obj)) {
    return mlx_scatter_args_nd(src, nb::cast<nb::tuple>(obj), vals);
  } else if (obj.is_none()) {
    return {{}, broadcast_to(vals, src.shape()), {}};
  } else if (nb::isinstance<nb::list>(obj)) {
    return mlx_scatter_args_array(
        src, array_from_list(nb::cast<nb::list>(obj), {}), vals);
  }

  throw std::invalid_argument("Cannot index mlx array using the given type.");
}

auto mlx_slice_update(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  if (src.ndim() == 0 ||
      (!nb::isinstance<nb::slice>(obj) && !nb::isinstance<nb::tuple>(obj))) {
    return std::make_pair(false, src);
  }
  if (nb::isinstance<nb::tuple>(obj)) {
    for (auto idx : nb::cast<nb::tuple>(obj)) {
      if (nb::isinstance<array>(idx) || nb::isinstance<nb::list>(idx)) {
        return std::make_pair(false, src);
      }
    }
  }

  auto upd = to_array(v, src.dtype());

  int s = 0;
  for (; s < upd.ndim() && upd.shape(s) == 1; s++) {
  };
  auto up_shape = std::vector<int>(upd.shape().begin() + s, upd.shape().end());
  up_shape = up_shape.empty() ? std::vector{1} : up_shape;
  auto up = reshape(upd, up_shape);

  std::vector<int> starts(src.ndim(), 0);
  std::vector<int> stops = src.shape();
  std::vector<int> strides(src.ndim(), 1);

  if (nb::isinstance<nb::slice>(obj)) {
    get_slice_params(
        starts[0],
        stops[0],
        strides[0],
        nb::cast<nb::slice>(obj),
        src.shape(0));

    auto out = slice_update(src, up, starts, stops, strides);
    return std::make_pair(true, out);
  }

  auto entries = nb::cast<nb::tuple>(obj);

  auto [non_none_indices, indices] = mlx_expand_ellipsis(src.shape(), entries);

  if (non_none_indices > src.ndim()) {
    std::ostringstream msg;
    msg << "Too many indices for array with " << src.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (non_none_indices == 0) {
    return std::make_pair(true, broadcast_to(up, src.shape()));
  }

  std::vector<int> up_reshape(src.ndim());
  int ax = src.ndim() - 1;
  int up_ax = up.ndim() - 1;
  for (; ax >= non_none_indices; ax--) {
    if (up_ax >= 0) {
      up_reshape[ax] = up.shape(up_ax);
      up_ax--;
    } else {
      up_reshape[ax] = 1;
    }
  }

  for (int i = indices.size() - 1; i >= 0; --i) {
    auto& pyidx = indices[i];
    if (nb::isinstance<nb::slice>(pyidx)) {
      get_slice_params(
          starts[ax],
          stops[ax],
          strides[ax],
          nb::cast<nb::slice>(pyidx),
          src.shape(ax));
      up_reshape[ax] = (up_ax >= 0) ? up.shape(up_ax--) : 1;
      ax--;
    } else if (nb::isinstance<nb::int_>(pyidx)) {
      int st = nb::cast<int>(pyidx);
      st = (st < 0) ? st + src.shape(ax) : st;
      starts[ax] = st;
      stops[ax] = st + 1;
      up_reshape[ax] = 1;
      ax--;
    }
  }

  up = reshape(up, std::move(up_reshape));
  auto out = slice_update(src, up, starts, stops, strides);
  return std::make_pair(true, out);
}

void mlx_set_item(array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [success, out] = mlx_slice_update(src, obj, v);
  if (success) {
    src.overwrite_descriptor(out);
    return;
  }

  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    auto out = scatter(src, indices, updates, axes);
    src.overwrite_descriptor(out);
  } else {
    src.overwrite_descriptor(updates);
  }
}

array mlx_add_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_add(src, indices, updates, axes);
  } else {
    return src + updates;
  }
}

array mlx_subtract_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_add(src, indices, -updates, axes);
  } else {
    return src - updates;
  }
}

array mlx_multiply_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_prod(src, indices, updates, axes);
  } else {
    return src * updates;
  }
}

array mlx_divide_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_prod(src, indices, reciprocal(updates), axes);
  } else {
    return src / updates;
  }
}

array mlx_maximum_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_max(src, indices, updates, axes);
  } else {
    return maximum(src, updates);
  }
}

array mlx_minimum_item(
    const array& src, const nb::object& obj, const ScalarOrArray& v) {
  auto [indices, updates, axes] = mlx_compute_scatter_args(src, obj, v);
  if (indices.size() > 0) {
    return scatter_min(src, indices, updates, axes);
  } else {
    return minimum(src, updates);
  }
}
