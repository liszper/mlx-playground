#include "mlx/backend/common/reduce.h"

namespace mlx::core {

std::pair<std::vector<int>, std::vector<size_t>> shapes_without_reduction_axes(
    const array& x,
    const std::vector<int>& axes) {
  std::vector<int> shape = x.shape();
  std::vector<size_t> strides = x.strides();

  for (int i = axes.size() - 1; i >= 0; i--) {
    int a = axes[i];
    shape.erase(shape.begin() + a);
    strides.erase(strides.begin() + a);
  }

  return std::make_pair(shape, strides);
}

ReductionPlan get_reduction_plan(const array& x, const std::vector<int>& axes) {
  if (x.size() == x.data_size() && axes.size() == x.ndim() &&
      x.flags().contiguous) {
    return ContiguousAllReduce;
  }

  if (x.flags().row_contiguous) {
    std::vector<int> shape = {x.shape(axes[0])};
    std::vector<size_t> strides = {x.strides()[axes[0]]};
    for (int i = 1; i < axes.size(); i++) {
      if (axes[i] - 1 == axes[i - 1] && x.shape(axes[i]) > 1) {
        shape.back() *= x.shape(axes[i]);
        strides.back() = x.strides()[axes[i]];
      } else {
        shape.push_back(x.shape(axes[i]));
        strides.push_back(x.strides()[axes[i]]);
      }
    }

    for (int i = shape.size() - 1; i >= 0; i--) {
      if (shape[i] == 1) {
        shape.erase(shape.begin() + i);
        strides.erase(strides.begin() + i);
      }
    }

    if (strides.back() == 1) {
      return ReductionPlan(ContiguousReduce, shape, strides);
    } else if (strides.back() > 1) {
      return ReductionPlan(ContiguousStridedReduce, shape, strides);
    }
  }

  std::vector<std::pair<int, size_t>> reductions;
  for (auto a : axes) {
    if (x.shape(a) > 1) {
      reductions.push_back(std::make_pair(x.shape(a), x.strides()[a]));
    }
  }
  std::sort(reductions.begin(), reductions.end(), [](auto a, auto b) {
    bool a_is_zero = a.second == 0;
    bool b_is_zero = b.second == 0;
    return (a_is_zero != b_is_zero) ? a.second < b.second : a.second > b.second;
  });
  for (int i = reductions.size() - 1; i >= 1; i--) {
    auto a = reductions[i];
    auto b = reductions[i - 1];

    if (b.second == a.first * a.second) {
      reductions.erase(reductions.begin() + i);
      reductions[i - 1] = std::make_pair(a.first * b.first, a.second);
    }
  }

  std::vector<int> shape;
  std::vector<size_t> strides;
  for (auto r : reductions) {
    shape.push_back(r.first);
    strides.push_back(r.second);
  }

  if (strides.back() == 1) {
    return ReductionPlan(GeneralContiguousReduce, shape, strides);
  }

  if (strides.back() > 1) {
    int size = 1;
    bool have_expand = false;
    for (int i = x.ndim() - 1; i >= 0; i--) {
      if (axes.back() == i) {
        continue;
      }

      size_t stride_i = x.strides()[i];
      int shape_i = x.shape(i);
      if (stride_i == 0) {
        if (shape_i == 1) {
          continue;
        }

        have_expand = true;
        break;
      }

      if (stride_i != size && shape_i != 1) {
        break;
      }
      size *= shape_i;
    }
    if (size > strides.back() || (size == strides.back() && !have_expand)) {
      return ReductionPlan(GeneralStridedReduce, shape, strides);
    }
  }

  return ReductionPlan(GeneralReduce, shape, strides);
}

}
