#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

inline auto collapse_batches(const array& a, const array& b) {
  std::vector<int> A_bshape{a.shape().begin(), a.shape().end() - 2};
  std::vector<int> B_bshape{b.shape().begin(), b.shape().end() - 2};
  if (A_bshape != B_bshape) {
    std::ostringstream msg;
    msg << "[matmul] Got matrices with incorrectly broadcasted shapes: " << "A "
        << a.shape() << ", B " << b.shape() << ".";
    throw std::runtime_error(msg.str());
  }

  std::vector<size_t> A_bstride{a.strides().begin(), a.strides().end() - 2};
  std::vector<size_t> B_bstride{b.strides().begin(), b.strides().end() - 2};

  auto [batch_shape, batch_strides] = collapse_contiguous_dims(A_bshape, std::vector{A_bstride, B_bstride});

  auto A_batch_stride = batch_strides[0];
  auto B_batch_stride = batch_strides[1];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    A_batch_stride.push_back(0);
    B_batch_stride.push_back(0);
  }

  return std::make_tuple(batch_shape, A_batch_stride, B_batch_stride);
}

}

void steel_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    std::vector<int> batch_shape /* = {} */,
    std::vector<size_t> A_batch_stride /* = {} */,
    std::vector<size_t> B_batch_stride /* = {} */) {
  using namespace mlx::steel;

  if (batch_shape.empty()) {
    auto [batch_shape_, A_bstride_, B_bstride_] = collapse_batches(a, b);

    batch_shape = batch_shape_;
    A_batch_stride = A_bstride_;
    B_batch_stride = B_bstride_;
    if (batch_size_out > 1 && !transpose_a && batch_shape.size() == 1 &&
        a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
        B_batch_stride.back() == 0) {
      M *= batch_shape.back();
      batch_size_out = 1;

      A_batch_stride = {0};
      B_batch_stride = {0};
      batch_shape = {1};
    }
  }

  size_t matrix_stride_out = size_t(M) * N;

  int _tm = M / 16;
  int _tn = N / 16;
  int _tk = K / 16;

  if (batch_size_out == 1 && (_tm * _tn) <= 32 && _tk >= 8) {
    int bm = M < 40 ? 16 : 32;
    int bn = N < 40 ? 16 : 32;
    int bk = 16;
    int wm = 2, wn = 2;

    int split_k_partitions = _tk < 16 ? 2 : (_tk < 32 ? 4 : (_tk < 64 ? 8 : 16));
    int split_k_partition_stride = M * N;
    int gemm_k_iterations = (K / bk) / split_k_partitions;
    int split_k_partition_size = gemm_k_iterations * bk;

    array C_split({split_k_partitions, M, N}, float32, nullptr, {});
    C_split.set_data(allocator::malloc_or_wait(C_split.nbytes()));
    copies.push_back(C_split);

    bool mn_aligned = M % bm == 0 && N % bn == 0;
    bool k_aligned = K % bk == 0;
    std::ostringstream kname;
    kname << "steel_gemm_splitk_" << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
          << type_to_name(C_split) << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn << "_MN_" << (mn_aligned ? "t" : "n")
          << "aligned" << "_K_" << (k_aligned ? "t" : "n") << "aligned";

    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = get_steel_gemm_splitk_kernel(
        d,
        kname.str(),
        a,
        C_split,
        transpose_a,
        transpose_b,
        bm,
        bn,
        bk,
        wm,
        wn,
        mn_aligned,
        k_aligned);
    compute_encoder->setComputePipelineState(kernel);

    int tn = (N + bn - 1) / bn;
    int tm = (M + bm - 1) / bm;

    GEMMSpiltKParams params{
        /* const int M = */ M,
        /* const int N = */ N,
        /* const int K = */ K,
        /* const int lda = */ lda,
        /* const int ldb = */ ldb,
        /* const int ldc = */ N,
        /* const int tiles_n = */ tn,
        /* const int tiles_m = */ tm,
        /* const int split_k_partitions = */ split_k_partitions,
        /* const int split_k_partition_stride = */ split_k_partition_stride,
        /* const int split_k_partition_size = */ split_k_partition_size,
        /* const int gemm_k_iterations_aligned = */ gemm_k_iterations};

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(tn, tm, split_k_partitions);

    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(b, 1);
    compute_encoder.set_output_array(C_split, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMSpiltKParams), 3);
    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    {
      auto c_split_buf = static_cast<const MTL::Resource*>(C_split.buffer().ptr());
      const class MTL::Resource* const resources[1] = {c_split_buf};
      compute_encoder->memoryBarrier(resources, 1);
      auto kernel_name = "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
          type_to_name(C_split);

      auto kernel = get_steel_gemm_splitk_accum_kernel(
          d, kernel_name, C_split, out, false);
      compute_encoder->setComputePipelineState(kernel);

      compute_encoder.set_input_array(C_split, 0);
      compute_encoder.set_output_array(out, 1);
      compute_encoder->setBytes(&split_k_partitions, sizeof(int), 2);
      compute_encoder->setBytes(&split_k_partition_stride, sizeof(int), 3);
      compute_encoder->setBytes(&N, sizeof(int), 4);

      MTL::Size grid_dims = MTL::Size(N, M, 1);
      MTL::Size group_dims = MTL::Size(std::min(1024, N * M), 1, 1);

      compute_encoder.dispatchThreads(grid_dims, group_dims);
    }

    if (!copies.empty()) {
      d.get_command_buffer(s.index)->addCompletedHandler(
          [copies = std::move(copies)](MTL::CommandBuffer*) mutable {
            copies.clear();
          });
    }
    return;
  }

  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  if ((size_t)batch_size_out * M * N >= 1ul << 20) {
    if (!transpose_a && transpose_b) {
      bm = 64;
      bn = (out.dtype() == float32) ? 64 : 32;
      bk = (out.dtype() == float32) ? 16 : 32;
    } else {
      bm = 64;
      bn = 64;
    }
  }

  std::ostringstream kname;
  kname << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn;

  std::string base_name = kname.str();

  const bool has_batch = (batch_shape.size() > 1);
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool do_gather = false;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
      {&do_gather, MTL::DataType::DataTypeBool, 300},
  };

  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_do_gather_" << (do_gather ? 't' : 'n');

  std::string hash_name = kname.str();

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn);

  compute_encoder->setComputePipelineState(kernel);

  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  int swizzle_log = 0;

  GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ A_batch_stride.back(),
      /* const size_t batch_stride_b = */ B_batch_stride.back(),
      /* const size_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  std::vector<size_t> batch_strides = A_batch_stride;
  batch_strides.insert(
      batch_strides.end(), B_batch_stride.begin(), B_batch_stride.end());

  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&params, sizeof(GEMMParams), 4);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  if (!copies.empty()) {
    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies = std::move(copies)](MTL::CommandBuffer*) mutable {
          copies.clear();
        });
  }
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error("[matmul] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    fill_gpu(zero, out, s);
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler([zero](MTL::CommandBuffer*) {});
    return;
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr, bool is_vector) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, a_cols, a] = check_transpose(a_pre, M == 1);
  auto [b_transposed, b_cols, b] = check_transpose(b_pre, N == 1);

  auto [batch_shape, A_batch_stride, B_batch_stride] = collapse_batches(a, b);

  auto batch_size_out = out.size() / (size_t(M) * size_t(N));

  if (batch_size_out > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
      B_batch_stride.back() == 0) {
    M *= batch_shape.back();
    batch_size_out = 1;

    A_batch_stride = {0};
    B_batch_stride = {0};
    batch_shape = {1};
  }

  if (std::min(M, N) == 1) {
    bool is_b_matrix = N != 1;

    auto& mat = is_b_matrix ? b : a;
    auto& vec = is_b_matrix ? a : b;
    bool transpose_mat = is_b_matrix ? !b_transposed : a_transposed;
    int in_vector_len = K;
    int out_vector_len = is_b_matrix ? N : M;

    int mat_cols = transpose_mat ? out_vector_len : in_vector_len;
    int mat_rows = transpose_mat ? in_vector_len : out_vector_len;
    int mat_ld = is_b_matrix ? b_cols : a_cols;

    auto batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
    auto batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;

    int stride_mat = batch_strides_mat.back();
    int stride_vec = batch_strides_vec.back();

    bool contiguous_kernel = (batch_shape.size() == 1);

    int batch_ndim = batch_shape.size();

    int tm = 4, tn = 4;
    int sm = 1, sn = 32;
    int bm = 1, bn = 1;
    int n_out_per_tgp;
    std::ostringstream kname;

    if (transpose_mat) {
      if (in_vector_len >= 8192 && out_vector_len >= 2048) {
        sm = 4;
        sn = 8;
      } else {
        sm = 8;
        sn = 4;
      }

      if (out_vector_len >= 2048) {
        bn = 16;
      } else if (out_vector_len >= 512) {
        bn = 4;
      } else {
        bn = 2;
      }

      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * sn * tn;
      kname << "gemv_t_" << type_to_name(out);

    } else {
      bm = out_vector_len >= 4096 ? 8 : 4;
      sn = 32;

      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * sm * tm;
      kname << "gemv_" << type_to_name(out);
    }

    kname << "_bm" << bm << "_bn" << bn << "_sm" << sm << "_sn" << sn << "_tm"
          << tm << "_tn" << tn;
    kname << "_nc" << !contiguous_kernel << "_axpby0";

    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(32, bn, bm);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    compute_encoder.set_input_array(mat, 0);
    compute_encoder.set_input_array(vec, 1);
    compute_encoder.set_output_array(out, 3);

    compute_encoder->setBytes(&in_vector_len, sizeof(int), 4);
    compute_encoder->setBytes(&out_vector_len, sizeof(int), 5);
    compute_encoder->setBytes(&mat_ld, sizeof(int), 6);

    compute_encoder->setBytes(&batch_ndim, sizeof(int), 9);
    set_vector_bytes(compute_encoder, batch_shape, 10);
    set_vector_bytes(compute_encoder, batch_strides_vec, 11);
    set_vector_bytes(compute_encoder, batch_strides_mat, 12);

    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    if (!copies.empty()) {
      d.get_command_buffer(s.index)->addCompletedHandler(
          [copies = std::move(copies)](MTL::CommandBuffer*) mutable {
            copies.clear();
          });
    }
    return;
  }

  return steel_matmul(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ a_cols,
      /* int ldb = */ b_cols,
      /* bool transpose_a = */ a_transposed,
      /* bool transpose_b = */ b_transposed,
      /* std::vector<array>& = */ copies,
      /* std::vector<int> batch_shape = */ batch_shape,
      /* std::vector<size_t> A_batch_stride = */ A_batch_stride,
      /* std::vector<size_t> B_batch_stride = */ B_batch_stride);
}

}
