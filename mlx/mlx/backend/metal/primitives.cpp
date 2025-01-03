#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/load.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/slicing.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core {

template <typename T>
void arange_set_scalars(T start, T next, CommandEncoder& enc) {
  enc->setBytes(&start, sizeof(T), 0);
  T step = next - start;
  enc->setBytes(&step, sizeof(T), 1);
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = get_arange_kernel(d, "arange" + type_to_name(out), out);
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(
      std::min(nthreads, kernel->maxTotalThreadsPerThreadgroup()), 1, 1);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  switch (out.dtype()) {
    case bool_:
      throw std::runtime_error("[Arange::eval_gpu] Does not support bool");
    case uint8:
      arange_set_scalars<uint8_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint16:
      arange_set_scalars<uint16_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint32:
      arange_set_scalars<uint32_t>(start_, start_ + step_, compute_encoder);
      break;
    case uint64:
      arange_set_scalars<uint64_t>(start_, start_ + step_, compute_encoder);
      break;
    case int8:
      arange_set_scalars<int8_t>(start_, start_ + step_, compute_encoder);
      break;
    case int16:
      arange_set_scalars<int16_t>(start_, start_ + step_, compute_encoder);
      break;
    case int32:
      arange_set_scalars<int32_t>(start_, start_ + step_, compute_encoder);
      break;
    case int64:
      arange_set_scalars<int64_t>(start_, start_ + step_, compute_encoder);
      break;
    case float16:
      arange_set_scalars<float16_t>(start_, start_ + step_, compute_encoder);
      break;
    case float32:
      arange_set_scalars<float>(start_, start_ + step_, compute_encoder);
      break;
    case bfloat16:
      arange_set_scalars<bfloat16_t>(start_, start_ + step_, compute_encoder);
      break;
  }

  compute_encoder.set_output_array(out, 2);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string op_name;
  switch (reduce_type_) {
    case ArgReduce::ArgMin:
      op_name = "argmin_";
      break;
    case ArgReduce::ArgMax:
      op_name = "argmax_";
      break;
  }

  std::vector<size_t> in_strides = in.strides();
  std::vector<int> shape = in.shape();
  std::vector<size_t> out_strides = out.strides();
  size_t axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];
  if (out_strides.size() == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  int simd_size = 32;
  int n_reads = 4;
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel(op_name + type_to_name(in));
    NS::UInteger thread_group_size = std::min(
        (axis_size + n_reads - 1) / n_reads,
        kernel->maxTotalThreadsPerThreadgroup());
    thread_group_size = (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    size_t n_threads = out.size() * thread_group_size;
    MTL::Size grid_dims = MTL::Size(n_threads, 1, 1);
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    if (ndim == 0) {
      int shape_ = 0;
      size_t stride_ = 0;
      compute_encoder->setBytes(&shape_, sizeof(int), 2);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 3);
      compute_encoder->setBytes(&stride_, sizeof(size_t), 4);
    } else {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 2);
      compute_encoder->setBytes(in_strides.data(), ndim * sizeof(size_t), 3);
      compute_encoder->setBytes(out_strides.data(), ndim * sizeof(size_t), 4);
    }
    compute_encoder->setBytes(&ndim, sizeof(size_t), 5);
    compute_encoder->setBytes(&axis_stride, sizeof(size_t), 6);
    compute_encoder->setBytes(&axis_size, sizeof(size_t), 7);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  CopyType ctype = inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  concatenate_gpu(inputs, out, axis_, stream());
}

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Depends::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto in = inputs[0];
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_gpu(in, out, ctype);
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto read_task = [out = out,
                    offset = offset_,
                    reader = reader_,
                    swap_endianness = swap_endianness_]() mutable {
    load(out, offset, reader, swap_endianness);
  };

  if (out.nbytes() > (1 << 28)) {
    read_task();
    return;
  }
  auto fut = io::thread_pool().enqueue(std::move(read_task)).share();
  auto signal_task = [out = out, fut = std::move(fut)]() {
    fut.wait();
    out.event().signal();
  };
  scheduler::enqueue(io_stream(), std::move(signal_task));
  auto& d = metal::device(stream().device);
  d.end_encoding(stream().index);
  auto command_buffer = d.get_command_buffer(stream().index);
  command_buffer->encodeWait(
      static_cast<MTL::Event*>(out.event().raw_event().get()),
      out.event().value());
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  assert(val.size() == 1);

  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  pad_gpu(in, val, out, axes_, low_pad_size_, stream());
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  size_t out_per_key = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_per_key / 2;
  bool odd = out_per_key % 2;

  auto& s = stream();
  auto& d = metal::device(s.device);
  std::string kname = keys.flags().row_contiguous ? "rbitsc" : "rbits";
  auto kernel = d.get_kernel(kname);

  MTL::Size grid_dims = MTL::Size(num_keys, half_size + odd, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(keys, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&odd, sizeof(bool), 2);
  compute_encoder->setBytes(&bytes_per_key, sizeof(size_t), 3);

  if (!keys.flags().row_contiguous) {
    int ndim = keys.ndim();
    compute_encoder->setBytes(&ndim, sizeof(int), 4);
    compute_encoder->setBytes(
        keys.shape().data(), keys.ndim() * sizeof(int), 5);
    compute_encoder->setBytes(
        keys.strides().data(), keys.ndim() * sizeof(size_t), 6);
  }

  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];

  auto [copy_necessary, out_strides] = prepare_reshape(in, out);

  if (copy_necessary) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    auto out_strides = make_contiguous_strides<size_t>(in.shape());
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        out_strides,
        0,
        0,
        CopyType::General,
        stream());
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void Split::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  slice_gpu(in, out, start_indices_, strides_, stream());
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  auto [data_offset, out_strides] = prepare_slice(out);

  std::vector<int64_t> upd_strides{upd.strides().begin(), upd.strides().end()};
  copy_gpu_inplace<int64_t>(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd_strides,
      /* const std::vector<stride_t>& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      /* const Stream& s = */ stream());
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[QRF::eval_gpu] Metal QR factorization NYI.");
}

void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto ibytes = size_of(in.dtype());
  auto obytes = size_of(out.dtype());
  if (ibytes == obytes || obytes < ibytes && in.strides().back() == 1 ||
      in.flags().row_contiguous) {
    auto strides = in.strides();
    for (int i = 0; i < strides.size() - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    out.copy_shared_buffer(
        in, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(in.shape(), in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc_or_wait(tmp.nbytes()));
    copy_gpu_inplace(in, tmp, CopyType::General, stream());

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.move_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

}
