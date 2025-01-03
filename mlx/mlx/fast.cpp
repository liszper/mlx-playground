#include <cassert>
#include <iostream>
#include <numeric>
#include <regex>

#include "mlx/backend/common/compiled.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (j < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

array rms_norm(
    const array& x,
    const array& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (weight.ndim() != 1) {
    std::ostringstream msg;
    msg << "[rms_norm] weight must have 1 dimension but has " << weight.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  auto out_type = result_type(x, weight);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [eps, out_type, s](const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);
    x = multiply(
        x,
        rsqrt(
            add(mean(square(x, s), -1, /* keepdims */ true, s),
                array(eps, float32),
                s),
            s),
        s);
    x = astype(x, out_type, s);
    return std::vector<array>{multiply(inputs[1], x, s)};
  };
  if (s.device == Device::gpu) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), astype(weight, out_type, s)});
  }
  return fallback({x, weight})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    vjps.push_back(
        sum(multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    int offset,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (inputs.size() == 2 &&
      (inputs[1].ndim() != 1 || inputs[1].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[1].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, offset, forward, s](
                      std::vector<array> inputs) {
    auto& shape = inputs[0].shape();
    int ndim = shape.size();
    auto x = reshape(inputs[0], {-1, shape[ndim - 2], shape[ndim - 1]}, s);
    auto t = x.dtype();
    auto N = x.shape(1) + offset;
    auto half_dims = dims / 2;
    auto positions = multiply(arange(offset, N, t, s), array(scale, t), s);

    auto default_inv_freqs = [&inputs, &s, &t, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, t, s),
              array(std::log(base) / half_dims, t),
              s),
          s);
    };

    auto inv_freqs = inputs.size() == 2 ? reciprocal(inputs[1], s) : default_inv_freqs();
    auto theta = multiply(expand_dims(positions, 1, s), expand_dims(inv_freqs, 0, s), s);
    auto coss = cos(theta, s);
    auto sins = sin(theta, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0}, {x.shape(0), x.shape(1), dims}, {1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 1}, {x.shape(0), x.shape(1), dims}, {1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, 3, s);
      }
      auto out = concatenate(outs, 3, s);
      if (dims < x.shape(-1)) {
        out = reshape(out, {x.shape(0), x.shape(1), dims});
        out = concatenate({out, slice(x, {0, 0, dims}, x.shape(), s)}, 2, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, 2, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (stream.device == Device::gpu) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, offset, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument("[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      offset,
      true,
      s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   offset = offset_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{rope(
        std::move(inputs),
        dims,
        traditional,
        base,
        scale,
        offset,
        !forward,
        s)};
  };

  auto inputs = cotangents;
  if (primals.size() == 2) {
    inputs.push_back(primals[1]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, offset_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      offset_ == a_other.offset_ && forward_ == a_other.forward_);
}

array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask,
    const std::optional<int> memory_efficient_threshold,
    StreamOrDevice s) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  /* generic implementation for use cases that Metal implementation does not
   * support. For non-supported cases listed below, use MLX primitives:
   * * CPU implementation
   * * batch size > 1 for decoding or causal attention
   * * query sequence length > 1 for decoding
   * * query sequence length > 16 && non-null mask (causal attention)
   * * non-null mask
   * * dtype is not fp32 or fp16
   */

  int threshold = 1e6;
  if (memory_efficient_threshold.has_value()) {
    threshold = std::max(1, memory_efficient_threshold.value());
  }

  bool needs_mask = mask.has_value();
  auto fallback = [scale, needs_mask, final_type, n_q_heads, n_kv_heads, &s](
                      const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    int B = q.shape(0);
    int L = q.shape(2);
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = reshape(q, {B, n_kv_heads, n_repeats, L, -1}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (needs_mask) {
      scores = add(scores, inputs[3], s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = reshape(out, {B, n_q_heads, L, -1}, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  const size_t query_head_dim = q.shape(-1);
  const bool supported_head_dim = query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128;

  const bool supported_head_dim_self_attn = query_head_dim == 64 || query_head_dim == 128;
  const size_t query_sequence_length = q.shape(2);
  const bool supports_full_self_attention = query_sequence_length >= 16 &&
      !mask.has_value() && supported_head_dim_self_attn &&
      n_q_heads == n_kv_heads && final_type != bfloat16 &&
      stream.device == Device::gpu;

  bool supports_sdpa = batch_dim == 1 && query_sequence_length == 1 &&
      !mask.has_value() && supported_head_dim && final_type != bfloat16 &&
      stream.device == Device::gpu;
  bool implementation_supports_use_case = supports_sdpa || supports_full_self_attention;

  const int seq_for_threshold = queries.shape(2);
  bool use_memory_efficient_impl = seq_for_threshold >= threshold;
  implementation_supports_use_case &= use_memory_efficient_impl;

  if (implementation_supports_use_case) {
    auto out_shape = std::vector<int>({q.shape(0), q.shape(1), q.shape(2), v.shape(-1)});
    auto out = array(
        std::move(out_shape),
        final_type,
        std::make_shared<ScaledDotProductAttention>(
            stream, fallback, scale, false),
        {q, k, v});
    return out;
  }

  if (mask.has_value()) {
    return fallback({q, k, v, mask.value()})[0];
  } else {
    return fallback({q, k, v})[0];
  }
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other = static_cast<const ScaledDotProductAttention&>(other);
  return needs_mask_ == a_other.needs_mask_ && scale_ == a_other.scale_;
}

}
