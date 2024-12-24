#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

std::vector<array> Primitive::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  std::ostringstream msg;
  msg << "[Primitive::jvp] Not implemented for ";
  print(msg);
  msg << ".";
  throw std::invalid_argument(msg.str());
}

std::vector<array> Primitive::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  std::ostringstream msg;
  msg << "[Primitive::vjp] Not implemented for ";
  print(msg);
  msg << ".";
  throw std::invalid_argument(msg.str());
}

std::vector<std::vector<int>> Primitive::output_shapes(
    const std::vector<array>&) {
  std::ostringstream msg;
  msg << "[Primitive::output_shapes] ";
  this->print(msg);
  msg << " cannot infer output shapes.";
  throw std::invalid_argument(msg.str());
}

std::vector<array> Abs::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Abs::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], sign(primals[0], stream()), stream())};
}

std::vector<array> Add::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {
      tangents.size() > 1 ? add(tangents[0], tangents[1], stream())
                          : tangents[0]};
}

std::vector<array> Add::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  if (argnums.size() == 1) {
    return cotangents;
  } else {
    return {cotangents[0], cotangents[0]};
  }
}

bool Arange::is_equivalent(const Primitive& other) const {
  const Arange& a_other = static_cast<const Arange&>(other);
  return (
      start_ == a_other.start_ && stop_ == a_other.stop_ &&
      step_ == a_other.step_);
}

bool ArgReduce::is_equivalent(const Primitive& other) const {
  const ArgReduce& r_other = static_cast<const ArgReduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axis_ == r_other.axis_;
}

std::vector<array> ArgReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {zeros_like(primals[0], stream())};
}

std::vector<array> ArgReduce::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {zeros_like(tangents[0], stream())};
}

std::vector<std::vector<int>> ArgReduce::output_shapes(
    const std::vector<array>& inputs) {
  auto out_shape = inputs[0].shape();
  out_shape[axis_] = 1;
  return {out_shape};
}

std::vector<array> AsType::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  if (cotangents[0].dtype() != dtype_) {
    throw std::invalid_argument("[astype] Type of cotangents does not match primal output type.");
  }
  return {astype(cotangents[0], primals[0].dtype(), stream())};
}

std::vector<array> AsType::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {astype(tangents[0], dtype_, stream())};
}

bool AsType::is_equivalent(const Primitive& other) const {
  const AsType& a_other = static_cast<const AsType&>(other);
  return dtype_ == a_other.dtype_;
}

std::vector<array> AsStrided::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(argnums.size() == 1);

  int grad_size = primals[0].size();
  int cotangents_size = cotangents[0].size();

  auto grad = zeros_like(primals[0], stream());
  grad = reshape(grad, {grad_size}, stream());

  auto idx = arange(grad_size, stream());
  idx = as_strided(idx, shape_, strides_, offset_, stream());
  idx = reshape(idx, {cotangents_size}, stream());

  auto flat_cotangents = reshape(cotangents[0], {cotangents_size, 1}, stream());

  grad = scatter_add(grad, idx, flat_cotangents, 0, stream());
  grad = reshape(grad, primals[0].shape(), stream());

  return {grad};
}

std::vector<array> AsStrided::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);

  return {as_strided(tangents[0], shape_, strides_, offset_, stream())};
}

bool AsStrided::is_equivalent(const Primitive& other) const {
  const AsStrided& a_other = static_cast<const AsStrided&>(other);
  return shape_ == a_other.shape_ && strides_ == a_other.strides_ &&
      offset_ == a_other.offset_;
}

bool BitwiseBinary::is_equivalent(const Primitive& other) const {
  const BitwiseBinary& a_other = static_cast<const BitwiseBinary&>(other);
  return op_ == a_other.op_;
}

void BitwiseBinary::print(std::ostream& os) {
  switch (op_) {
    case BitwiseBinary::And:
      os << "BitwiseAnd";
      break;
    case BitwiseBinary::Or:
      os << "BitwiseOr";
      break;
    case BitwiseBinary::Xor:
      os << "BitwiseXor";
      break;
    case BitwiseBinary::LeftShift:
      os << "LeftShift";
      break;
    case BitwiseBinary::RightShift:
      os << "RightShift";
      break;
  }
}

std::vector<array> BitwiseBinary::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(tangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> BitwiseBinary::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Broadcast::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(argnums.size() == 1);

  auto& shape = primals[0].shape();
  auto& cotan = cotangents[0];
  int diff = cotan.ndim() - shape.size();
  std::vector<int> reduce_axes;
  for (int i = 0; i < cotan.ndim(); ++i) {
    if (i < diff) {
      reduce_axes.push_back(i);
    } else if (shape[i - diff] != cotan.shape(i)) {
      reduce_axes.push_back(i);
    }
  }
  return {reshape(sum(cotan, reduce_axes, true, stream()), shape, stream())};
}

std::vector<array> Broadcast::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1);
  return {broadcast_to(tangents[0], shape_, stream())};
}

bool Broadcast::is_equivalent(const Primitive& other) const {
  const Broadcast& b_other = static_cast<const Broadcast&>(other);
  return shape_ == b_other.shape_;
}

std::vector<array> Ceil::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Ceil::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::vector<array> Concatenate::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& cotan = cotangents[0];
  std::vector<int> start(cotan.ndim(), 0);
  std::vector<int> stop = cotan.shape();

  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : primals) {
    sizes.push_back(p.shape(axis_));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  std::vector<array> grads;
  for (auto i : argnums) {
    start[axis_] = sizes[i];
    stop[axis_] = sizes[i + 1];
    grads.push_back(slice(cotan, start, stop, stream()));
  }
  return grads;
}

std::vector<array> Concatenate::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<int> argidx(argnums.size());
  std::iota(argidx.begin(), argidx.end(), 0);
  std::sort(argidx.begin(), argidx.end(), [&argnums](int a, int b) {
    return argnums[a] < argnums[b];
  });

  std::vector<array> vals;
  for (int i = 0, j = 0; i < primals.size(); ++i) {
    if (j < argnums.size() && argnums[argidx[j]] == i) {
      vals.push_back(tangents[argidx[j++]]);
    } else {
      vals.push_back(zeros_like(primals[i], stream()));
    }
  }
  return {concatenate(vals, axis_, stream())};
}

bool Concatenate::is_equivalent(const Primitive& other) const {
  const Concatenate& c_other = static_cast<const Concatenate&>(other);
  return axis_ == c_other.axis_;
}

std::vector<array> Copy::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return cotangents;
}

std::vector<array> Copy::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents;
}

std::vector<array> Cos::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return {jvp(primals, cotangents, argnums)};
}

std::vector<array> Cos::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(
      tangents[0], negative(sin(primals[0], stream()), stream()), stream())};
}

std::vector<array> Cosh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Cosh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], sinh(primals[0], stream()), stream())};
}

std::vector<array> Depends::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  std::vector<array> vjps;

  for (auto arg : argnums) {
    if (arg < cotangents.size()) {
      vjps.push_back(cotangents[arg]);
    } else {
      vjps.push_back(zeros_like(primals[arg]));
    }
  }
  return vjps;
}

std::vector<array> Divide::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(divide(cotangents[0], primals[1], stream()));
    } else {
      vjps.push_back(negative(
          divide(
              multiply(cotangents[0], primals[0], stream()),
              square(primals[1], stream()),
              stream()),
          stream()));
    }
  }
  return vjps;
}

std::vector<array> DivMod::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> DivMod::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {zeros_like(primals[0], stream())};
}

std::vector<array> Divide::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    if (arg == 0) {
      return divide(tangents[i], primals[1], stream());
    } else {
      return negative(
          divide(
              multiply(tangents[i], primals[0], stream()),
              square(primals[1], stream()),
              stream()),
          stream());
    }
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return {out};
}

std::vector<array> Remainder::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(cotangents[0]);
    } else {
      auto x_over_y = divide(primals[0], primals[1], stream());
      x_over_y = floor(x_over_y, stream());
      vjps.push_back(
          negative(multiply(x_over_y, cotangents[0], stream()), stream()));
    }
  }
  return vjps;
}

std::vector<array> Remainder::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    if (arg == 0) {
      return tangents[i];
    } else {
      auto x_over_y = divide(primals[0], primals[1], stream());
      x_over_y = floor(x_over_y, stream());
      return negative(multiply(x_over_y, tangents[i], stream()), stream());
    }
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return {out};
}

std::vector<array> Equal::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Equal::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Erf::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Erf::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(M_2_SQRTPI, dtype), tangents[0], stream());
  return {multiply(
      scale,
      exp(negative(square(primals[0], stream()), stream()), stream()),
      stream())};
}

std::vector<array> ErfInv::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(1.0 / M_2_SQRTPI, dtype), cotangents[0], stream());
  return {
      multiply(scale, exp(square(outputs[0], stream()), stream()), stream())};
}

std::vector<array> ErfInv::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(1.0 / M_2_SQRTPI, dtype), tangents[0], stream());
  return {multiply(
      scale,
      exp(square(erfinv(primals[0], stream()), stream()), stream()),
      stream())};
}

std::vector<array> Exp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return {multiply(cotangents[0], outputs[0], stream())};
}

std::vector<array> Exp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], exp(primals[0], stream()), stream())};
}

std::vector<array> Expm1::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return {multiply(
      cotangents[0],
      add(outputs[0], array(1.0f, outputs[0].dtype()), stream()),
      stream())};
}

std::vector<array> Expm1::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], exp(primals[0], stream()), stream())};
}

std::vector<array> Floor::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Floor::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::vector<array> Full::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(cotangents[0], primals[0], stream())};
}

std::vector<array> Full::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents;
}

std::vector<array> Gather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (int argnum : argnums) {
    if (argnum > 0) {
      vjps.push_back(
          zeros(primals[argnum].shape(), primals[argnum].dtype(), stream()));
    } else {
      auto src = zeros_like(primals[0], stream());
      std::vector<array> inds(primals.begin() + 1, primals.end());
      vjps.push_back(scatter_add(src, inds, cotangents[0], axes_, stream()));
    }
  }
  return vjps;
}

std::vector<array> Gather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument("[gather] Cannot calculate JVP with respect to indices.");
  }
  std::vector<array> inds(primals.begin() + 1, primals.end());
  return {gather(tangents[0], inds, axes_, slice_sizes_, stream())};
}

bool Gather::is_equivalent(const Primitive& other) const {
  const Gather& g_other = static_cast<const Gather&>(other);
  return axes_ == g_other.axes_ && slice_sizes_ == g_other.slice_sizes_;
}

std::vector<array> Greater::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Greater::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> GreaterEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> GreaterEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Less::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Less::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> LessEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> LessEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Log::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Log::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto out = divide(tangents[0], primals[0], stream());
  if (base_ != Base::e) {
    auto scale = 1 / std::log(base_ == Base::ten ? 10.0f : 2.0f);
    out = multiply(array(scale, out.dtype()), out, stream());
  }
  return {out};
}

std::vector<array> Log1p::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Log1p::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  return {divide(
      tangents[0], add(array(1.0f, dtype), primals[0], stream()), stream())};
}

std::vector<array> LogicalNot::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> LogicalNot::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(tangents[0], stream())};
}

std::vector<array> LogicalAnd::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(cotangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> LogicalAnd::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(argnums.size() <= 2);
  return {zeros_like(primals[0], stream())};
}

std::vector<array> LogicalOr::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(cotangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> LogicalOr::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(argnums.size() <= 2);

  return {zeros_like(primals[0], stream())};
}

std::vector<array> Matmul::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  auto& cotan = cotangents[0];
  std::vector<int> reorder(cotan.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  std::iter_swap(reorder.end() - 1, reorder.end() - 2);
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(
          matmul(cotan, transpose(primals[1], reorder, stream()), stream()));
    } else {
      vjps.push_back(
          matmul(transpose(primals[0], reorder, stream()), cotan, stream()));
    }
  }
  return vjps;
}

std::vector<array> Maximum::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask = (arg == 0) ? greater(a, b, stream()) : less_equal(a, b, stream());
    vjps.push_back(multiply(cotangents[0], mask, stream()));
  }
  return {vjps};
}

std::vector<array> Maximum::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    auto mask = (arg == 0) ? greater(a, b, stream()) : less_equal(a, b, stream());
    return multiply(tangents[i], mask, stream());
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return {out};
}

std::vector<array> Minimum::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask = (arg == 0) ? less(a, b, stream()) : greater_equal(a, b, stream());
    vjps.push_back(multiply(cotangents[0], mask, stream()));
  }
  return vjps;
}

std::vector<array> Minimum::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    auto mask = (arg == 0) ? less(a, b, stream()) : greater_equal(a, b, stream());
    return multiply(tangents[i], mask, stream());
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return {out};
}

std::vector<array> Multiply::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto arg = argnums[0];
  auto jvp = multiply(tangents[0], primals[1 - arg], stream());
  if (argnums.size() > 1) {
    arg = argnums[1];
    jvp = add(jvp, multiply(tangents[1], primals[1 - arg], stream()), stream());
  }
  return {jvp};
}

std::vector<array> Multiply::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(primals[1 - arg], cotangents[0], stream()));
  }
  return vjps;
}

std::vector<array> Select::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 3);
  assert(tangents.size() == 3);

  auto jvp_fun = [&](int i) {
    int arg = argnums[i];

    if (arg == 0) {
      return zeros_like(primals[0], stream());
    } else if (arg == 1) {
      return multiply(
          astype(primals[0], tangents[1].dtype(), stream()),
          tangents[1],
          stream());
    } else {
      return multiply(
          astype(
              logical_not(primals[0], stream()), tangents[2].dtype(), stream()),
          tangents[2],
          stream());
    }
  };

  array jvp = jvp_fun(argnums[0]);
  for (int i = 1; i < argnums.size(); i++) {
    jvp = add(jvp, jvp_fun(argnums[i]));
  }
  return {jvp};
}

std::vector<array> Select::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 3);
  assert(cotangents.size() == 1);

  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(zeros_like(primals[0], stream()));
    } else if (arg == 1) {
      vjps.push_back(multiply(
          astype(primals[0], cotangents[0].dtype(), stream()),
          cotangents[0],
          stream()));
    } else if (arg == 2) {
      vjps.push_back(multiply(
          astype(
              logical_not(primals[0], stream()),
              cotangents[0].dtype(),
              stream()),
          cotangents[0],
          stream()));
    }
  }
  return vjps;
}

std::vector<array> Negative::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Negative::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {negative(tangents[0], stream())};
}

std::vector<array> NotEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> NotEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Pad::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  auto& cotan = cotangents[0];
  std::vector<int> start(cotan.ndim(), 0);
  std::vector<int> stop = cotan.shape();

  for (auto i : axes_) {
    start[i] = low_pad_size_[i];
    stop[i] -= high_pad_size_[i];
  }

  auto out = slice(cotan, start, stop, stream());

  return {out};
}

std::vector<array> Pad::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  return {
      pad(tangents[0],
          axes_,
          low_pad_size_,
          high_pad_size_,
          array(0, tangents[0].dtype()),
          "constant",
          stream())};
}

bool Pad::is_equivalent(const Primitive& other) const {
  const Pad& p_other = static_cast<const Pad&>(other);
  return (
      p_other.axes_ == axes_ && p_other.low_pad_size_ == low_pad_size_ &&
      p_other.high_pad_size_ == high_pad_size_);
}

std::vector<array> Power::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(multiply(
          power(
              primals[0],
              subtract(primals[1], array(1, primals[0].dtype()), stream()),
              stream()),
          primals[1],
          stream()));
    } else {
      auto& exp = outputs[0];
      auto exp_vjp = multiply(log(primals[0], stream()), outputs[0], stream());
      vjps.push_back(where(exp, exp_vjp, array(0.0f, exp.dtype()), stream()));
    }
    vjps.back() = multiply(cotangents[0], vjps.back(), stream());
  }
  return vjps;
}

std::vector<array> Power::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto output = power(primals[0], primals[1], stream());
  auto grads = vjp(primals, tangents, argnums, {output});
  if (argnums.size() > 1) {
    return {add(grads[0], grads[1], stream())};
  } else {
    return grads;
  }
}

bool RandomBits::is_equivalent(const Primitive& other) const {
  const RandomBits& r_other = static_cast<const RandomBits&>(other);
  return shape_ == r_other.shape_;
}

std::vector<array> Reshape::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return {reshape(cotangents[0], primals[0].shape(), stream())};
}

std::vector<array> Reshape::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return {reshape(tangents[0], shape_, stream())};
}

bool Reshape::is_equivalent(const Primitive& other) const {
  const Reshape& r_other = static_cast<const Reshape&>(other);
  return shape_ == r_other.shape_;
}

std::vector<array> Reduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto in = primals[0];

  std::vector<int> shape = in.shape();
  for (auto ax : axes_) {
    shape[ax] = 1;
  }
  auto& cotan = cotangents[0];
  if (reduce_type_ == Reduce::Sum) {
    return {
        broadcast_to(reshape(cotan, shape, stream()), in.shape(), stream())};
  } else if (reduce_type_ == Reduce::Prod) {
    auto s = stream();
    auto prod_grad_single_axis =
        [&s](const array& x, const array& cotan, int axis) {
          auto p1 = cumprod(x, axis, /*reverse=*/false, /*inclusive=*/false, s);
          auto p2 = cumprod(x, axis, /*reverse=*/true, /*inclusive=*/false, s);
          auto exclusive_prod = multiply(p1, p2, s);
          return multiply(exclusive_prod, cotan, s);
        };

    if (axes_.size() > 1) {
      std::vector<int> transpose_to;
      std::vector<int> transpose_back;
      std::vector<int> shape_flat;
      {
        int j = 0;
        for (int i = 0; i < in.ndim(); i++) {
          if (j < axes_.size() && axes_[j] == i) {
            j++;
          } else {
            transpose_to.push_back(i);
            shape_flat.push_back(in.shape(i));
          }
        }
        for (auto ax : axes_) {
          transpose_to.push_back(ax);
        }
        shape_flat.push_back(-1);
        transpose_back.resize(transpose_to.size());
        for (int i = 0; i < transpose_to.size(); i++) {
          transpose_back[transpose_to[i]] = i;
        }
      }

      auto x = transpose(in, transpose_to, s);
      auto shape_to = x.shape();

      x = reshape(x, shape_flat, stream());
      auto grad = prod_grad_single_axis(x, reshape(cotan, shape_flat, s), -1);

      grad = reshape(grad, shape_to, s);
      grad = transpose(grad, transpose_back, s);

      return {grad};
    } else {
      return {prod_grad_single_axis(in, reshape(cotan, shape, s), axes_[0])};
    }

  } else if (reduce_type_ == Reduce::Min || reduce_type_ == Reduce::Max) {
    auto out = outputs[0];
    if (out.ndim() != in.ndim()) {
      out = expand_dims(out, axes_, stream());
    }
    auto mask = equal(in, out, stream());
    auto normalizer = sum(mask, axes_, true, stream());
    auto cotan_reshape = reshape(cotan, shape, stream());
    cotan_reshape = divide(cotan_reshape, normalizer, stream());
    return {multiply(cotan_reshape, mask, stream())};
  }

  else {
    throw std::runtime_error("Reduce type VJP not yet implemented.");
  }
}

bool Reduce::is_equivalent(const Primitive& other) const {
  const Reduce& r_other = static_cast<const Reduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axes_ == r_other.axes_;
}

std::vector<std::vector<int>> Reduce::output_shapes(
    const std::vector<array>& inputs) {
  std::vector<int> out_shape = inputs[0].shape();
  for (auto i : axes_) {
    out_shape[i] = 1;
  }
  return {out_shape};
}

std::vector<array> Round::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Round::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::vector<array> Scan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return {cumsum(cotangents[0], axis_, !reverse_, inclusive_, stream())};
  } else if (reduce_type_ == Scan::Prod) {
    auto in = primals[0];

    auto cprod_exclusive = cumprod(in, axis_, reverse_, !inclusive_, stream());
    auto cprod_inclusive = outputs[0];
    if (!inclusive_) {
      std::swap(cprod_exclusive, cprod_inclusive);
    }

    auto z = array(0, in.dtype());
    auto eq_zero = equal(cprod_inclusive, z, stream());
    auto first_zero = logical_and(eq_zero, not_equal(cprod_exclusive, z, stream()), stream());

    auto to_partial_grad = [this, &cotangents](const array& arr) {
      return cumsum(
          multiply(arr, cotangents[0], stream()),
          axis_,
          !reverse_,
          inclusive_,
          stream());
    };

    auto cprod_with_one = cumprod(
        where(first_zero, array(1, in.dtype()), in, stream()),
        axis_,
        reverse_,
        inclusive_,
        stream());
    auto grad_with_one = to_partial_grad(cprod_with_one);
    auto grad = divide(to_partial_grad(outputs[0]), in, stream());
    return {where(
        first_zero,
        grad_with_one,
        where(eq_zero, z, grad, stream()),
        stream())};
  } else {
    throw std::runtime_error("VJP is not implemented for cumulative min/max");
  }
}

std::vector<array> Scan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(tangents.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return {cumsum(tangents[0], axis_, reverse_, inclusive_, stream())};
  } else {
    throw std::runtime_error("JVP is not implemented for cumulative prod/min/max");
  }
}

bool Scan::is_equivalent(const Primitive& other) const {
  const Scan& s_other = static_cast<const Scan&>(other);
  return (
      reduce_type_ == s_other.reduce_type_ && axis_ == s_other.axis_ &&
      reverse_ == s_other.reverse_ && inclusive_ == s_other.inclusive_);
}

bool Scatter::is_equivalent(const Primitive& other) const {
  const Scatter& s_other = static_cast<const Scatter&>(other);
  return reduce_type_ == s_other.reduce_type_ && axes_ == s_other.axes_;
}

std::vector<array> Scatter::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  switch (reduce_type_) {
    case Scatter::None:
    case Scatter::Sum:
    case Scatter::Max:
    case Scatter::Min:
      break;
    default:
      throw std::runtime_error("[scatter] VJP not implemented for scatter_prod");
  }

  const array& result = outputs[0];
  const array& values = primals[0];
  const array& updates = primals.back();
  const std::vector<array> indices(primals.begin() + 1, primals.end() - 1);

  std::vector<array> vjps;
  for (auto num : argnums) {
    if (num == 0) {
      switch (reduce_type_) {
        case Scatter::None:
          vjps.push_back(scatter(
              cotangents[0],
              indices,
              zeros_like(updates, stream()),
              axes_,
              stream()));
          break;
        case Scatter::Sum:
          vjps.push_back(cotangents[0]);
          break;
        case Scatter::Max:
        case Scatter::Min: {
          vjps.push_back(where(
              equal(result, values, stream()),
              cotangents[0],
              array(0, cotangents[0].dtype()),
              stream()));
          break;
        }
        default:
          throw std::invalid_argument("");
      }
    } else if (num == primals.size() - 1) {
      switch (reduce_type_) {
        case Scatter::None:
        case Scatter::Sum: {
          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          vjps.push_back(
              gather(cotangents[0], indices, axes_, slice_sizes, stream()));
          break;
        }
        case Scatter::Max:
        case Scatter::Min: {
          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          auto gathered_cotan = gather(cotangents[0], indices, axes_, slice_sizes, stream());
          auto gathered_result = gather(result, indices, axes_, slice_sizes, stream());
          vjps.push_back(
              multiply(gathered_cotan, gathered_result == updates, stream()));
          break;
        }
        default: {
          throw std::invalid_argument("");
        }
      }
    } else {
      throw std::invalid_argument("[scatter] Cannot calculate VJP with respect to indices.");
    }
  }
  return vjps;
}

std::vector<array> Scatter::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("[scatter] JVP not yet implemented");
}

std::vector<array> Sigmoid::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto& s = outputs[0];
  auto sprime = multiply(s, subtract(array(1.0f, s.dtype()), s, stream()), stream());
  return {multiply(cotangents[0], sprime, stream())};
}

std::vector<array> Sigmoid::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto s = sigmoid(primals[0], stream());
  auto sprime = multiply(s, subtract(array(1.0f, s.dtype()), s, stream()), stream());
  return {multiply(tangents[0], sprime, stream())};
}

std::vector<array> Sign::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sign::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros(primals[0].shape(), primals[0].dtype(), stream())};
}

std::vector<array> Sin::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sin::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], cos(primals[0], stream()), stream())};
}

std::vector<array> Sinh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sinh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], cosh(primals[0], stream()), stream())};
}

std::vector<array> Slice::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);

  std::vector<array> inds;
  std::vector<int> ind_axes;
  std::vector<array> single_inds;
  std::vector<int> single_ind_axes;
  for (int i = 0; i < start_indices_.size(); ++i) {
    auto start = start_indices_[i];
    auto end = end_indices_[i];
    auto stride = strides_[i];
    if (start == 0 && stride == 1) {
      continue;
    }
    if (stride == 1) {
      single_inds.push_back(array(start));
      single_ind_axes.push_back(i);
    } else {
      inds.push_back(arange(start, end, stride, stream()));
      ind_axes.push_back(i);
    }
  }

  auto cotan = cotangents[0];
  if (!ind_axes.empty()) {
    std::vector<int> cotan_shape;
    for (auto ax : ind_axes) {
      cotan_shape.push_back(cotan.shape(ax));
    }
    std::vector<int> cotan_axes(ind_axes);
    for (int j = 0, i = 0; i < cotan.ndim(); ++i) {
      if (j < ind_axes.size() && ind_axes[j] == i) {
        cotan_shape.push_back(1);
        j++;
      } else {
        cotan_shape.push_back(cotan.shape(i));
        cotan_axes.push_back(i);
      }
    }
    cotan = reshape(transpose(cotan, cotan_axes, stream()), cotan_shape, stream());
  }

  std::vector<int> inds_shape(inds.size(), 1);
  for (int i = 0; i < inds.size(); ++i) {
    inds_shape[i] = inds[i].size();
    inds[i] = reshape(inds[i], inds_shape, stream());
    inds_shape[i] = 1;
  }

  inds.insert(inds.end(), single_inds.begin(), single_inds.end());
  ind_axes.insert(
      ind_axes.end(), single_ind_axes.begin(), single_ind_axes.end());

  return {scatter_add(
      zeros_like(primals[0], stream()), inds, cotan, ind_axes, stream())};
}

std::vector<array> Slice::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  return {slice(tangents[0], start_indices_, end_indices_, strides_, stream())};
}

bool Slice::is_equivalent(const Primitive& other) const {
  const Slice& s_other = static_cast<const Slice&>(other);
  return (
      start_indices_ == s_other.start_indices_ &&
      end_indices_ == s_other.end_indices_ && strides_ == s_other.strides_);
}

std::vector<array> SliceUpdate::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);

  auto& cotan = cotangents[0];
  auto& src = primals[0];
  auto& upd = primals[1];

  std::vector<array> vjps;

  for (int num : argnums) {
    if (num == 0) {
      auto grad = slice_update(
          cotan,
          zeros_like(upd, stream()),
          start_indices_,
          end_indices_,
          strides_,
          stream());

      vjps.push_back(grad);
    }
    else {
      auto grad = slice(cotan, start_indices_, end_indices_, strides_, stream());

      vjps.push_back(grad);
    }
  }

  return vjps;
}

std::vector<array> SliceUpdate::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  return {slice_update(
      tangents[0],
      tangents[1],
      start_indices_,
      end_indices_,
      strides_,
      stream())};
}

bool SliceUpdate::is_equivalent(const Primitive& other) const {
  const SliceUpdate& s_other = static_cast<const SliceUpdate&>(other);
  return (
      start_indices_ == s_other.start_indices_ &&
      end_indices_ == s_other.end_indices_ && strides_ == s_other.strides_);
}

std::vector<array> Softmax::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(cotangents.size() == 1);
  auto& s = outputs[0];
  auto sv = multiply(s, cotangents[0], stream());
  return {subtract(
      sv,
      multiply(s, sum(sv, std::vector<int>{-1}, true, stream()), stream()))};
}

std::vector<array> Softmax::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto s = softmax(primals[0], std::vector<int>{-1}, precise_, stream());
  auto sv = multiply(s, tangents[0], stream());
  return {subtract(
      sv,
      multiply(s, sum(sv, std::vector<int>{-1}, true, stream()), stream()))};
}

bool Softmax::is_equivalent(const Primitive& other) const {
  const Softmax& s_other = static_cast<const Softmax&>(other);
  return precise_ == s_other.precise_;
}

std::vector<array> Split::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return {concatenate(cotangents, axis_, stream())};
}

std::vector<array> Split::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return split(tangents[0], indices_, axis_, stream());
}

bool Split::is_equivalent(const Primitive& other) const {
  const Split& s_other = static_cast<const Split&>(other);
  return axis_ == s_other.axis_ && indices_ == s_other.indices_;
}

std::vector<array> Square::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Square::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return {multiply(
      primals[0],
      multiply(array(2, primals[0].dtype()), tangents[0], stream()),
      stream())};
}

std::vector<array> Sqrt::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(cotangents.size() == 1);
  auto dtype = primals[0].dtype();
  if (recip_) {
    auto one_over_x_root_x = divide(outputs[0], primals[0], stream());
    return {multiply(
        multiply(array(-0.5, dtype), cotangents[0], stream()),
        one_over_x_root_x,
        stream())};
  } else {
    return {divide(
        multiply(array(0.5, dtype), cotangents[0], stream()),
        outputs[0],
        stream())};
  }
}

std::vector<array> Sqrt::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  if (recip_) {
    return vjp(primals, tangents, argnums, {rsqrt(primals[0], stream())});
  } else {
    return vjp(primals, tangents, argnums, {sqrt(primals[0], stream())});
  }
}

bool Sqrt::is_equivalent(const Primitive& other) const {
  const Sqrt& s_other = static_cast<const Sqrt&>(other);
  return recip_ == s_other.recip_;
}

std::vector<array> Subtract::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto vjp = cotangents[0];
    if (arg == 1) {
      vjp = negative(vjp, stream());
    }
    vjps.push_back(vjp);
  }
  return vjps;
}

std::vector<array> Subtract::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    return arg == 1 ? negative(tangents[i], stream()) : tangents[i];
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return {out};
}

std::vector<array> Tan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Tan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cos_sq = square(cos(primals[0], stream()), stream());
  return {divide(tangents[0], cos_sq, stream())};
}

std::vector<array> Tanh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Tanh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cosh_sq = square(cosh(primals[0], stream()), stream());
  return {divide(tangents[0], cosh_sq, stream())};
}

std::vector<array> Transpose::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  std::vector<int> iaxes(axes_.size());
  for (int i = 0; i < axes_.size(); ++i) {
    iaxes[axes_[i]] = i;
  }
  return {transpose(cotangents[0], iaxes, stream())};
}

std::vector<array> Transpose::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return {transpose(tangents[0], axes_, stream())};
}

bool Transpose::is_equivalent(const Primitive& other) const {
  const Transpose& t_other = static_cast<const Transpose&>(other);
  return axes_ == t_other.axes_;
}

bool NumberOfElements::is_equivalent(const Primitive& other) const {
  const NumberOfElements& n_other = static_cast<const NumberOfElements&>(other);
  return axes_ == n_other.axes_ && inverted_ == n_other.inverted_ &&
      dtype_ == n_other.dtype_;
}

void View::print(std::ostream& os) {
  os << "View" << dtype_;
}

bool View::is_equivalent(const Primitive& other) const {
  const View& a_other = static_cast<const View&>(other);
  return (dtype_ == a_other.dtype_);
}

}
