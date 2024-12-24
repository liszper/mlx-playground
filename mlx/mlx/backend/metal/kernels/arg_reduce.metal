#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr constant U init = Limits<U>::max;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr constant U init = Limits<U>::min;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
IndexValPair<U> simd_shuffle_down(IndexValPair<U> data, uint16_t delta) {
  return IndexValPair<U>{
      simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)};
}

template <typename T, typename Op, int N_READS = 4>
[[kernel]] void arg_reduce_general(
    const device T* in [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    const constant int* shape [[buffer(2)]],
    const constant size_t* in_strides [[buffer(3)]],
    const constant size_t* out_strides [[buffer(4)]],
    const constant size_t& ndim [[buffer(5)]],
    const constant size_t& axis_stride [[buffer(6)]],
    const constant size_t& axis_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  auto in_idx = elem_to_loc(gid / lsize, shape, in_strides, ndim);
  auto out_idx = elem_to_loc(gid / lsize, shape, out_strides, ndim);

  IndexValPair<T> best{0, Op::init};

  threadgroup IndexValPair<T> local_data[32];

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize); r++) {
    uint32_t current_index = r * lsize * N_READS + lid * N_READS;
    uint32_t offset = current_index;
    const device T* current_in = in + in_idx + current_index * axis_stride;
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (current_index < axis_size) ? *current_in : T(Op::init);
      current_index++;
      current_in += axis_stride;
    }
    best = op.template reduce_many<N_READS>(best, vals, offset);
  }

  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }

  uint simd_groups = ceildiv(lsize, simd_size);
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  }
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  if (lid == 0) {
    out[out_idx] = best.index;
  }
}

#define instantiate_arg_reduce(name, itype)                      \
  instantiate_kernel(                                            \
      "argmin_" #name, arg_reduce_general, itype, ArgMin<itype>) \
  instantiate_kernel(                                            \
      "argmax_" #name, arg_reduce_general, itype, ArgMax<itype>)

instantiate_arg_reduce(bool_, bool)
instantiate_arg_reduce(uint32, uint32_t)
instantiate_arg_reduce(uint64, uint64_t)
instantiate_arg_reduce(int32, int32_t)
instantiate_arg_reduce(int64, int64_t)
instantiate_arg_reduce(float32, float)
instantiate_arg_reduce(bfloat16, bfloat16_t)
