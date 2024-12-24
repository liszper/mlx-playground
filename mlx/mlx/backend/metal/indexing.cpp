#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/jit/indexing.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

constexpr int METAL_MAX_INDEX_ARRAYS = 20;

std::pair<std::string, std::string> make_index_args(
    const std::string& idx_type,
    int nidx) {
  std::ostringstream idx_args;
  std::ostringstream idx_arr;
  for (int i = 0; i < nidx; ++i) {
    idx_args << fmt::format("const device {0} *idx{1} [[buffer({2})]],", idx_type, i, 20 + i);
    idx_arr << fmt::format("idx{0}", i);
    if (i < nidx - 1) {
      idx_args << "\n";
      idx_arr << ",";
    }
  }
  return {idx_args.str(), idx_arr.str()};
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& src = inputs[0];
  int nidx = inputs.size() - 1;

  if (nidx > METAL_MAX_INDEX_ARRAYS) {
    std::ostringstream msg;
    msg << "[Gather::eval_gpu] Gathering with more than " << METAL_MAX_INDEX_ARRAYS << " index arrays not yet supported.";
    throw std::runtime_error(msg.str());
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  size_t ndim = src.ndim();

  std::string lib_name;
  std::string kernel_name;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  {
    std::ostringstream kname;
    kname << "gather" << type_to_name(out) << idx_type_name << "_" << nidx << "_" << idx_ndim;
    lib_name = kname.str();
    kernel_name = lib_name;
  }

  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gather();
    std::string out_type_str = get_type_string(out.dtype());
    std::string idx_type_str = nidx ? get_type_string(inputs[1].dtype()) : "bool";
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    kernel_source << fmt::format(
        gather_kernels,
        type_to_name(out) + idx_type_name,
        out_type_str,
        idx_type_str,
        nidx,
        idx_args,
        idx_arr,
        idx_ndim);
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);
  compute_encoder->setComputePipelineState(kernel);

  size_t slice_size = 1;
  for (auto s : slice_sizes_) {
    slice_size *= s;
  }

  size_t dim0 = 1;
  size_t dim1 = 1;
  if (nidx) {
    if (inputs[1].ndim() >= 1) {
      dim0 = inputs[1].shape(0);
    }
    if (inputs[1].ndim() >= 2) {
      dim1 = inputs[1].size() / dim0;
    }
  }
  size_t dim2 = slice_size;
  auto group_dims = get_block_dims(dim0, dim1, dim2);
  MTL::Size grid_dims = MTL::Size(dim0, dim1, dim2);

  std::vector<int> idx_shapes;
  std::vector<size_t> idx_strides;

  for (int i = 0; i < nidx; ++i) {
    idx_shapes.insert(
        idx_shapes.end(),
        inputs[i + 1].shape().begin(),
        inputs[i + 1].shape().end());

    idx_strides.insert(
        idx_strides.end(),
        inputs[i + 1].strides().begin(),
        inputs[i + 1].strides().end());
  }

  compute_encoder.set_input_array(src, 0);
  compute_encoder.set_output_array(out, 1);

  compute_encoder->setBytes(src.shape().data(), ndim * sizeof(int), 2);
  compute_encoder->setBytes(src.strides().data(), ndim * sizeof(size_t), 3);
  compute_encoder->setBytes(&ndim, sizeof(size_t), 4);
  compute_encoder->setBytes(slice_sizes_.data(), ndim * sizeof(int), 5);
  compute_encoder->setBytes(axes_.data(), nidx * sizeof(int), 6);

  compute_encoder->setBytes(
      idx_shapes.data(), idx_shapes.size() * sizeof(int), 7);
  compute_encoder->setBytes(
      idx_strides.data(), idx_strides.size() * sizeof(size_t), 8);
  compute_encoder->setBytes(&idx_ndim, sizeof(int), 9);

  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], 20 + i);
  }

  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (size_of(out.dtype()) == 8) {
    std::ostringstream msg;
    msg << "[Scatter::eval_gpu] Does not support " << out.dtype();
    throw std::invalid_argument(msg.str());
  }

  int nidx = axes_.size();
  if (nidx > METAL_MAX_INDEX_ARRAYS) {
    std::ostringstream msg;
    msg << "[Scatter::eval_gpu] Gathering with more than "
        << METAL_MAX_INDEX_ARRAYS << " index arrays not yet supported.";
    throw std::runtime_error(msg.str());
  }

  auto copy_type = inputs[0].data_size() == 1 ? CopyType::Scalar : CopyType::General;
  copy_gpu(inputs[0], out, copy_type);

  if (inputs.back().size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  bool index_nd1_specialization = (idx_ndim == 1);

  for (auto i = 0; i < axes_.size() && index_nd1_specialization; i++) {
    index_nd1_specialization &= (axes_[i] == i);
  }

  for (int i = 1; i < inputs.size() && index_nd1_specialization; i++) {
    index_nd1_specialization &= inputs[i].flags().row_contiguous;
  }

  std::string lib_name;
  std::string kernel_name;
  std::string idx_type_name = nidx ? type_to_name(inputs[1]) : "";
  std::string op_name;
  switch (reduce_type_) {
    case Scatter::None:
      op_name = "none";
      break;
    case Scatter::Sum:
      op_name = "sum";
      break;
    case Scatter::Prod:
      op_name = "prod";
      break;
    case Scatter::Max:
      op_name = "max";
      break;
    case Scatter::Min:
      op_name = "min";
      break;
  }

  {
    std::ostringstream kname;
    if (index_nd1_specialization) {
      kname << "scatter_1d_index" << type_to_name(out) << idx_type_name;
    } else {
      kname << "scatter" << type_to_name(out) << idx_type_name;
    }
    kname << "_" << op_name << "_" << nidx;
    lib_name = kname.str();
    kernel_name = kname.str();
  }

  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::reduce_utils() << metal::scatter();

    std::string out_type_str = get_type_string(out.dtype());
    std::string idx_type_str = nidx ? get_type_string(inputs[1].dtype()) : "bool";
    std::string op_type;
    switch (reduce_type_) {
      case Scatter::None:
        op_type = "None";
        break;
      case Scatter::Sum:
        op_type = "Sum";
        break;
      case Scatter::Prod:
        op_type = "Prod";
        break;
      case Scatter::Max:
        op_type = "Max";
        break;
      case Scatter::Min:
        op_type = "Min";
        break;
    }
    if (reduce_type_ != Scatter::None) {
      std::ostringstream oss;
      oss << op_type << "<" << out_type_str << ">" ;
      op_type = oss.str();
    }
    auto [idx_args, idx_arr] = make_index_args(idx_type_str, nidx);

    kernel_source << fmt::format(
        scatter_kernels,
        type_to_name(out) + idx_type_name + "_" + op_name,
        out_type_str,
        idx_type_str,
        op_type,
        nidx,
        idx_args,
        idx_arr);
    lib = d.get_library(lib_name, kernel_source.str());
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, lib);

  auto& upd = inputs.back();
  size_t nthreads = upd.size();

  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(upd, 1);
  compute_encoder.set_output_array(out, 2);

  uint upd_ndim = upd.ndim();
  size_t upd_size = 1;
  for (int i = idx_ndim; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  if (index_nd1_specialization) {
    compute_encoder->setBytes(
        out.shape().data(), out.shape().size() * sizeof(int), 3);
    compute_encoder->setBytes(
        out.strides().data(), out.strides().size() * sizeof(size_t), 4);

    size_t out_ndim = out.ndim();
    compute_encoder->setBytes(&out_ndim, sizeof(out_ndim), 5);
    if (upd_ndim <= 1) {
      int shape_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 6);
    } else {
      compute_encoder->setBytes(upd.shape().data(), upd_ndim * sizeof(int), 6);
    }
    compute_encoder->setBytes(&upd_ndim, sizeof(size_t), 7);
    compute_encoder->setBytes(&upd_size, sizeof(size_t), 8);

    for (int i = 0; i < nidx; ++i) {
      compute_encoder.set_input_array(inputs[i + 1], 20 + i);
    }

    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

  } else {
    std::vector<int> idx_shapes;
    std::vector<size_t> idx_strides;

    for (int i = 0; i < nidx; ++i) {
      idx_shapes.insert(
          idx_shapes.end(),
          inputs[i + 1].shape().begin(),
          inputs[i + 1].shape().end());

      idx_strides.insert(
          idx_strides.end(),
          inputs[i + 1].strides().begin(),
          inputs[i + 1].strides().end());
    }

    if (upd_ndim == 0) {
      int shape_ = 0;
      size_t stride_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 3);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 4);
    } else {
      compute_encoder->setBytes(upd.shape().data(), upd_ndim * sizeof(int), 3);
      compute_encoder->setBytes(
          upd.strides().data(), upd_ndim * sizeof(size_t), 4);
    }
    compute_encoder->setBytes(&upd_ndim, sizeof(size_t), 5);
    compute_encoder->setBytes(&upd_size, sizeof(size_t), 6);

    size_t out_ndim = out.ndim();
    if (out_ndim == 0) {
      int shape_ = 0;
      size_t stride_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 7);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 8);
    } else {
      compute_encoder->setBytes(out.shape().data(), out_ndim * sizeof(int), 7);
      compute_encoder->setBytes(
          out.strides().data(), out_ndim * sizeof(size_t), 8);
    }
    compute_encoder->setBytes(&out_ndim, sizeof(size_t), 9);
    compute_encoder->setBytes(axes_.data(), axes_.size() * sizeof(int), 10);

    if (idx_ndim == 0) {
      idx_shapes.push_back(0);
      idx_strides.push_back(0);
    }
    compute_encoder->setBytes(
        idx_shapes.data(), idx_shapes.size() * sizeof(int), 11);
    compute_encoder->setBytes(
        idx_strides.data(), idx_strides.size() * sizeof(size_t), 12);
    compute_encoder->setBytes(&idx_ndim, sizeof(int), 13);

    for (int i = 0; i < nidx; ++i) {
      compute_encoder.set_input_array(inputs[i + 1], 20 + i);
    }

    MTL::Size grid_dims = MTL::Size(upd_size, nthreads / upd_size, 1);
    MTL::Size group_dims = get_block_dims(upd_size, nthreads / upd_size, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

}
