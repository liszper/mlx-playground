#include <algorithm>
#include <cassert>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

struct RowReduceArgs {
  std::vector<int> shape;
  std::vector<size_t> strides;
  int ndim;

  std::vector<int> reduce_shape;
  std::vector<size_t> reduce_strides;
  int reduce_ndim;

  size_t non_row_reductions;

  size_t row_size;

  RowReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    row_size = plan.shape.back();

    reduce_shape = plan.shape;
    reduce_strides = plan.strides;
    reduce_shape.pop_back();
    reduce_strides.pop_back();
    reduce_ndim = reduce_shape.size();

    non_row_reductions = 1;
    for (auto s : reduce_shape) {
      non_row_reductions *= s;
    }

    std::tie(shape, strides) = shapes_without_reduction_axes(in, axes);
    std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
    ndim = shape.size();
  }

  void encode(CommandEncoder& compute_encoder) {
    if (reduce_ndim == 0) {
      reduce_shape.push_back(0);
      reduce_strides.push_back(0);
    }
    if (ndim == 0) {
      shape.push_back(0);
      strides.push_back(0);
    }

    compute_encoder->setBytes(&row_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 3);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 5);
    compute_encoder->setBytes(&ndim, sizeof(int), 6);
    compute_encoder->setBytes(
        reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
    compute_encoder->setBytes(
        reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
    compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);

    if (reduce_ndim == 0) {
      reduce_shape.pop_back();
      reduce_strides.pop_back();
    }
    if (ndim == 0) {
      shape.pop_back();
      strides.pop_back();
    }
  }
};

struct ColReduceArgs {
  std::vector<int> shape;
  std::vector<size_t> strides;
  int ndim;

  std::vector<int> reduce_shape;
  std::vector<size_t> reduce_strides;
  int reduce_ndim;

  size_t non_col_reductions;

  size_t reduction_size;
  size_t reduction_stride;

  ColReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    reduction_size = plan.shape.back();
    reduction_stride = plan.strides.back();

    reduce_shape = plan.shape;
    reduce_strides = plan.strides;
    reduce_shape.pop_back();
    reduce_strides.pop_back();
    reduce_ndim = reduce_shape.size();

    non_col_reductions = 1;
    for (auto s : reduce_shape) {
      non_col_reductions *= s;
    }

    size_t stride_back = 1;
    std::tie(shape, strides) = shapes_without_reduction_axes(in, axes);
    while (!shape.empty() && stride_back < reduction_stride) {
      stride_back *= shape.back();
      shape.pop_back();
      strides.pop_back();
    }
    std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
    ndim = shape.size();
  }

  void encode(CommandEncoder& compute_encoder) {
    if (reduce_ndim == 0) {
      reduce_shape.push_back(0);
      reduce_strides.push_back(0);
    }
    if (ndim == 0) {
      shape.push_back(0);
      strides.push_back(0);
    }

    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 5);
    compute_encoder->setBytes(&ndim, sizeof(int), 6);
    compute_encoder->setBytes(
        reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
    compute_encoder->setBytes(
        reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
    compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);
    compute_encoder->setBytes(&non_col_reductions, sizeof(size_t), 10);

    if (reduce_ndim == 0) {
      reduce_shape.pop_back();
      reduce_strides.pop_back();
    }
    if (ndim == 0) {
      shape.pop_back();
      strides.pop_back();
    }
  }
};

}

inline auto safe_div(size_t n, size_t m) {
  return m == 0 ? 0 : (n + m - 1) / m;
}

inline auto safe_divup(size_t n, size_t m) {
  return safe_div(n, m) * m;
}

inline bool is_64b_int(Dtype dtype) {
  return dtype == int64 || dtype == uint64;
}

inline bool is_64b_dtype(Dtype dtype) {
  return dtype == int64 || dtype == uint64;
}

inline int threadgroup_size_from_row_size(int row_size) {
  if (row_size <= 512) {
    return 32;
  }

  if (row_size <= 1024) {
    return 128;
  }

  int thread_group_size;
  thread_group_size = (row_size + REDUCE_N_READS - 1) / REDUCE_N_READS;
  thread_group_size = ((thread_group_size + 31) / 32) * 32;
  thread_group_size = std::min(1024, thread_group_size);
  return thread_group_size;
}

inline auto output_grid_for_col_reduce(
    const array& out,
    const ColReduceArgs& args) {
  auto out_shape = out.shape();
  auto out_strides = out.strides();
  while (!out_shape.empty() && out_strides.back() < args.reduction_stride) {
    out_shape.pop_back();
    out_strides.pop_back();
  }
  return get_2d_grid_dims(out_shape, out_strides);
}

void init_reduce(
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto kernel = get_reduce_init_kernel(
      d, "init_reduce_" + op_name + type_to_name(out), out);
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_output_array(out, 0);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s,
    std::vector<array>& copies) {
  std::ostringstream kname;
  const std::string func_name = "all_reduce";
  kname << func_name << "_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  size_t in_size = in.size();

  if (in_size <= REDUCE_N_READS * 1024) {
    int threadgroup_size = (in_size + REDUCE_N_READS - 1) / REDUCE_N_READS;
    threadgroup_size = ((threadgroup_size + 31) / 32) * 32;
    MTL::Size grid_dims(threadgroup_size, 1, 1);

    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 3);
    compute_encoder.dispatchThreads(grid_dims, grid_dims);
  }

  else {
    int n_rows, threadgroup_2nd_pass;
    if (in.nbytes() <= (1 << 26)) {
      n_rows = 32 * REDUCE_N_READS;
      threadgroup_2nd_pass = 32;
    }

    else {
      n_rows = 1024 * REDUCE_N_READS;
      threadgroup_2nd_pass = 1024;
    }

    array intermediate({n_rows}, out.dtype(), nullptr, {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    copies.push_back(intermediate);

    size_t row_size = (in_size + n_rows - 1) / n_rows;
    int threadgroup_size = std::min((row_size + REDUCE_N_READS - 1) / REDUCE_N_READS, 1024ul);
    threadgroup_size = ((threadgroup_size + 31) / 32) * 32;
    MTL::Size grid_dims(threadgroup_size, n_rows, 1);
    MTL::Size group_dims(threadgroup_size, 1, 1);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(intermediate, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&row_size, sizeof(size_t), 3);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

    std::ostringstream kname_2nd_pass;
    kname_2nd_pass << func_name << "_" << op_name << type_to_name(intermediate);
    auto kernel_2nd_pass = get_reduce_kernel(
        d, kname_2nd_pass.str(), func_name, op_name, intermediate, out);
    compute_encoder->setComputePipelineState(kernel_2nd_pass);
    size_t intermediate_size = n_rows;
    grid_dims = MTL::Size(threadgroup_2nd_pass, 1, 1);
    group_dims = MTL::Size(threadgroup_2nd_pass, 1, 1);
    compute_encoder.set_input_array(intermediate, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&intermediate_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&intermediate_size, sizeof(size_t), 3);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

void row_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  std::ostringstream kname;
  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  const std::string func_name = "row_reduce_small";
  kname << func_name << "_" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out, n);
  compute_encoder->setComputePipelineState(kernel);

  MTL::Size grid_dims;
  MTL::Size group_dims;
  if ((args.non_row_reductions < 32 && args.row_size <= 8) ||
      args.non_row_reductions <= 8) {
    grid_dims = get_2d_grid_dims(out.shape(), out.strides());
    group_dims = MTL::Size((grid_dims.width < 1024) ? grid_dims.width : 1024, 1, 1);
  } else {
    auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
    grid_dims = MTL::Size(32, out_grid_size.width, out_grid_size.height);
    group_dims = MTL::Size(32, 1, 1);
  }

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_simple(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  std::ostringstream kname;
  const std::string func_name = "row_reduce_simple";
  kname << func_name << "_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  size_t row_size = args.row_size;
  size_t out_size = out.size();
  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  out_grid_size.width = (out_grid_size.width + REDUCE_N_WRITES - 1) / REDUCE_N_WRITES;
  int threadgroup_size = threadgroup_size_from_row_size(row_size);
  if (in.itemsize() == 8) {
    threadgroup_size = std::min(threadgroup_size, 512);
  }
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&row_size, sizeof(size_t), 2);
  compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  std::ostringstream kname;
  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  const std::string func_name = "row_reduce_looped";
  kname << func_name << "_" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out, n);
  compute_encoder->setComputePipelineState(kernel);

  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  int threadgroup_size = threadgroup_size_from_row_size(args.row_size);
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  RowReduceArgs args(in, plan, axes);

  if (args.row_size <= 64) {
    return row_reduce_small(in, out, op_name, args, compute_encoder, d, s);
  }

  if (plan.type == ContiguousReduce && args.reduce_ndim == 0 &&
      in.size() / args.row_size >= 32) {
    return row_reduce_simple(in, out, op_name, args, compute_encoder, d, s);
  }

  return row_reduce_looped(in, out, op_name, args, compute_encoder, d, s);
}

void strided_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  MTL::Size grid_dims, group_dims;

  if (args.reduction_size * args.non_col_reductions < 64 &&
      args.reduction_stride < 32) {
    grid_dims = output_grid_for_col_reduce(out, args);
    int threadgroup_size = (grid_dims.width > 128) ? 128 : grid_dims.width;
    group_dims = MTL::Size(threadgroup_size, 1, 1);
  }

  else if (args.reduction_size * args.non_col_reductions < 32) {
    auto out_grid_dims = output_grid_for_col_reduce(out, args);
    int threads_x = (args.reduction_stride + REDUCE_N_READS - 1) / REDUCE_N_READS;
    int threadgroup_x = std::min(threads_x, 128);
    grid_dims = MTL::Size(threads_x, out_grid_dims.width, out_grid_dims.height);
    group_dims = MTL::Size(threadgroup_x, 1, 1);
  }

  else {
    args.reduce_shape.push_back(args.reduction_size);
    args.reduce_strides.push_back(args.reduction_stride);
    args.reduce_ndim++;
    int simdgroups = (args.reduction_stride + REDUCE_N_READS - 1) / REDUCE_N_READS;
    int threadgroup_size = simdgroups * 32;
    auto out_grid_dims = output_grid_for_col_reduce(out, args);
    grid_dims = MTL::Size(threadgroup_size, out_grid_dims.width, out_grid_dims.height);
    group_dims = MTL::Size(threadgroup_size, 1, 1);
  }

  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  std::ostringstream kname;
  const std::string func_name = "col_reduce_small";
  kname << func_name << "_" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out, n);
  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void strided_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  auto out_grid_size = output_grid_for_col_reduce(out, args);
  int BN = (args.reduction_stride <= 1024) ? 32 : 128;
  int BM = 1024 / BN;
  int threadgroup_size = 4 * 32;
  MTL::Size grid_dims(
      threadgroup_size * ((args.reduction_stride + BN - 1) / BN),
      out_grid_size.width,
      out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  std::ostringstream kname;
  const std::string func_name = "col_reduce_looped";
  kname << func_name << "_" << n << "_" << BM << "_" << BN << "_reduce_"
        << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), func_name, op_name, in, out, n, BM, BN);
  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  ColReduceArgs args(in, plan, axes);

  if (args.reduction_stride < 32 ||
      args.reduction_size * args.non_col_reductions < 32) {
    return strided_reduce_small(in, out, op_name, args, compute_encoder, d, s);
  }

  return strided_reduce_looped(in, out, op_name, args, compute_encoder, d, s);
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  assert(!axes_.empty());
  assert(out.size() != in.size());

  size_t min_bytes = std::max(out.nbytes(), 4ul);
  out.set_data(allocator::malloc_or_wait(min_bytes));
  std::string op_name;
  switch (reduce_type_) {
    case Reduce::And:
      op_name = "and";
      break;
    case Reduce::Or:
      op_name = "or";
      break;
    case Reduce::Sum:
      op_name = "sum";
      break;
    case Reduce::Prod:
      op_name = out.dtype() == bool_ ? "and" : "prod";
      break;
    case Reduce::Min:
      op_name = out.dtype() == bool_ ? "and" : "min";
      break;
    case Reduce::Max:
      op_name = out.dtype() == bool_ ? "or" : "max";
      break;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s.index);

  if (in.size() > 0) {
    std::vector<array> copies;
    ReductionPlan plan = get_reduction_plan(in, axes_);

    if (plan.type == GeneralReduce) {
      array in_copy(in.shape(), in.dtype(), nullptr, {});
      copy_gpu(in, in_copy, CopyType::General, s);
      copies.push_back(in_copy);
      in = in_copy;
      plan = get_reduction_plan(in, axes_);
    }

    if (plan.type == ContiguousAllReduce) {
      all_reduce_dispatch(in, out, op_name, compute_encoder, d, s, copies);
    }

    else if (
        plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
      row_reduce_general_dispatch(
          in, out, op_name, plan, axes_, compute_encoder, d, s);
    }

    else if (
        plan.type == ContiguousStridedReduce ||
        plan.type == GeneralStridedReduce) {
      strided_reduce_general_dispatch(
          in, out, op_name, plan, axes_, compute_encoder, d, s);
    }

    if (!copies.empty()) {
      d.get_command_buffer(s.index)->addCompletedHandler(
          [copies = std::move(copies)](MTL::CommandBuffer*) mutable {
            copies.clear();
          });
    }
  }

  else {
    init_reduce(out, op_name, compute_encoder, d, s);
  }
}

}
