template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* inputs[N_WRITES],
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  Op op;

  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  int index = lid_x * N_READS;
  if (index + N_READS <= extra) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  } else {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; index + i < extra; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const constant size_t& reduction_size,
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const size_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }

    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC void
thread_reduce(thread U& total, const device T* row, int blocks, int extra) {
  Op op;
  for (int i = 0; i < blocks; i++) {
    U vals[N_READS];
    for (int j = 0; j < N_READS; j++) {
      vals[j] = row[j];
    }
    for (int j = 0; j < N_READS; j++) {
      total = op(vals[j], total);
    }
    row += N_READS;
  }
  for (int i = 0; i < extra; i++) {
    total = op(*row++, total);
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& row_size [[buffer(2)]],
    const constant size_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;

  U total_val = Op::init;
  looped_elem_to_loc<NDIMS> loop;

  const device T* row;
  int blocks = row_size / N_READS;
  int extra = row_size % N_READS;

  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    size_t out_idx = tid.x + tsize.y * size_t(tid.y);
    in += elem_to_loc(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }

    out[out_idx] = total_val;
  } else {
    size_t out_idx = gid.y + gsize.y * size_t(gid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim);

    loop.next(simd_lane_id, reduce_shape, reduce_strides);

    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    total_val = op.simd_reduce(total_val);

    if (simd_lane_id == 0) {
      out[out_idx] = total_val;
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];

  size_t out_idx = N_WRITES * (gid.y + gsize.y * size_t(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * reduction_size;
  out += out_idx;

  int blocks = reduction_size / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);

  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  if (lid.x == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      out[i] = totals[i];
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& row_size [[buffer(2)]],
    const constant size_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;

  size_t out_idx = gid.y + gsize.y * size_t(gid.z);

  in += elem_to_loc(out_idx, shape, strides, ndim) + lid.x * N_READS;

  looped_elem_to_loc<NDIMS> loop;
  const device T* row;
  int blocks = row_size / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);

  for (size_t i = 0; i < non_row_reductions; i++) {
    row = in + loop.location(i, reduce_shape, reduce_strides, reduce_ndim);

    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);

    total = op(total, row_total);

    loop.next(reduce_shape, reduce_strides);
  }

  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  if (lid.x == 0) {
    out[out_idx] = total;
  }
}
