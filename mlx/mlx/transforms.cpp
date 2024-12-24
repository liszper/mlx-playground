#include <algorithm>
#include <future>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/metal/metal_impl.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

class Synchronizer : public Primitive {
 public:
  explicit Synchronizer(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {}
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {}

  DEFINE_PRINT(Synchronize);
};

int detail::InTracing::tracing_counter{0};
int detail::RetainGraph::tracing_counter{0};

array eval_impl(std::vector<array> outputs, bool async) {
  std::queue<array> tape;

  std::unordered_map<uint32_t, Event> events;

  Stream stream = default_stream(default_device());
  for (auto& o : outputs) {
    if (o.status() == array::Status::unscheduled && o.has_primitive()) {
      stream = o.primitive().stream();
      break;
    }
  }

  std::unordered_set<uintptr_t> needs_signal;

  auto synchronizer = array(
      {}, bool_, std::make_shared<Synchronizer>(stream), std::move(outputs));
  needs_signal.insert(synchronizer.id());

  events.emplace(stream.index, Event{stream});

  {
    std::unordered_set<std::uintptr_t> cache;
    std::stack<std::pair<std::reference_wrapper<array>, int>> dfs;
    dfs.emplace(synchronizer, 0);
    while (!dfs.empty()) {
      auto& [a_ref, idx] = dfs.top();
      auto& a = a_ref.get();
      if (idx < a.inputs().size()) {
        auto& in = a.inputs()[idx++];

        if (in.status() == array::Status::scheduled) {
          continue;
        }

        if (!in.is_available()) {
          if (async && in.is_tracer()) {
            throw std::invalid_argument("[async_eval] Not allowed inside a graph transformation.");
          }
          if (!in.has_primitive()) {
            if (in.is_tracer()) {
              throw std::invalid_argument("[eval] Attempting to eval an array during function"
                  " transformations like compile or vmap is not allowed.");
            }
            throw std::runtime_error("[eval] Attempting to eval an array without a primitive.\n"
                "If you are compiling a function, make sure all the inputs "
                "and outputs are captured:\n"
                "https://ml-explore.github.io/mlx/build/html/usage/compile.html#pure-functions.\n"
                "If you are not using compile, this may be a bug. "
                "Please file an issue here:\n"
                "https://github.com/ml-explore/mlx/issues.");
          }
          if (a.primitive().stream() != in.primitive().stream()) {
            needs_signal.insert(in.id());
          }
        }

        if (cache.find(in.id()) == cache.end()) {
          dfs.emplace(in, 0);
          cache.insert(in.id());
          for (auto& s : in.siblings()) {
            cache.insert(s.id());
          }
        }
        continue;
      }

      if (a.is_available() && !a.is_tracer() && a.has_primitive()) {
        a.detach();
      } else if (a.status() == array::Status::unscheduled) {
        tape.push(a);
        auto& stream = a.primitive().stream();
        auto e = events.find(stream.index);
        if (e == events.end()) {
          e = events.emplace(stream.index, Event{stream}).first;
        }
        e->second.set_value(e->second.value() + 1);
        a.attach_event(e->second);
        for (auto& s : a.siblings()) {
          s.attach_event(e->second);
        }
      }
      dfs.pop();
    }
  }

  while (!tape.empty()) {
    auto arr = std::move(tape.front());
    tape.pop();

    arr.set_status(array::Status::scheduled);
    for (auto& s : arr.siblings()) {
      s.set_status(array::Status::scheduled);
    }

    auto stream = arr.primitive().stream();
    std::vector<std::shared_future<void>> arr_deps;
    bool signal = needs_signal.find(arr.id()) != needs_signal.end();

    if (arr.primitive().device() == Device::gpu) {
      if (!metal::is_available()) {
        throw std::runtime_error("Metal GPU is not available.");
      }
      scheduler::enqueue(stream, metal::make_task(std::move(arr), signal));
    } else {
      auto task = [arr = std::move(arr), stream, signal]() mutable {
        for (auto& input : arr.inputs()) {
          if (input.event().valid() &&
              input.event().stream() != arr.primitive().stream()) {
            input.event().wait();
          }
        }
        scheduler::notify_new_task(stream);
        auto outputs = arr.outputs();
        arr.primitive().eval_cpu(arr.inputs(), outputs);
        if (!arr.is_tracer()) {
          arr.detach();
        }
        for (auto& out : outputs) {
          out.set_status(array::Status::available);
        }

        if (signal) {
          arr.event().signal();
        }

        scheduler::notify_task_completion(stream);
      };
      scheduler::enqueue(stream, std::move(task));
    }
  }
  return synchronizer;
}

void async_eval(std::vector<array> outputs) {
  if (outputs.empty()) {
    return;
  }

  if (std::none_of(outputs.begin(), outputs.end(), [](array& x) {
        return x.status() == array::Status::unscheduled;
      })) {
    return;
  }

  eval_impl(std::move(outputs), true);
}

void eval(std::vector<array> outputs) {
  if (outputs.empty()) {
    return;
  }

  if (std::none_of(outputs.begin(), outputs.end(), [](array& x) {
        return x.status() == array::Status::unscheduled;
      })) {
    for (auto& x : outputs) {
      if (!x.is_available()) {
        x.event().wait();
      }
    }
    return;
  }

  eval_impl(std::move(outputs), false).event().wait();
}

std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotans) {
  detail::InTracing in_tracing;

  std::vector<array> primals_;
  for (auto& p : primals) {
    auto s = p.has_primitive() ? p.primitive().stream()
                               : default_stream(default_device());
    primals_.push_back(copy(p, s));
    primals_.back().set_tracer(true);
  }

  auto outputs = fun(primals_);

  int cotan_index = 0;
  std::vector<std::pair<int, int>> output_cotan_pairs;
  for (int i = 0; i < outputs.size(); ++i) {
    auto& out = outputs[i];
    if (out.has_primitive()) {
      if (auto& p = out.primitive(); typeid(p) == typeid(StopGradient)) {
        continue;
      }
    }
    if (cotan_index >= cotans.size()) {
      std::ostringstream msg;
      msg << "[vjp] Number of outputs to compute gradients for ("
          << outputs.size() << ") does not match number of cotangents ("
          << cotans.size() << ").";
      throw std::invalid_argument(msg.str());
    }
    if (out.shape() != cotans[cotan_index].shape()) {
      std::ostringstream msg;
      msg << "[vjp] Output shape " << out.shape()
          << " does not match cotangent shape " << cotans[cotan_index].shape()
          << ".";
      if (outputs.size() == 1 && out.size() == 1) {
        msg << " If you are using grad your function must return a scalar.";
      }
      throw std::invalid_argument(msg.str());
    }
    output_cotan_pairs.emplace_back(i, cotan_index++);
  }

  std::unordered_set<std::uintptr_t> cache;
  std::unordered_set<std::uintptr_t> calc_grad;
  for (auto& primal : primals_) {
    primal.set_tracer(false);
    calc_grad.insert(primal.id());
    cache.insert(primal.id());
  }

  std::vector<array> tape;

  std::function<void(array&)> recurse;
  recurse = [&](auto& a) {
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }
    a.set_tracer(false);
    for (auto& s : a.siblings()) {
      s.set_tracer(false);
      cache.insert(s.id());
    }

    for (auto& input : a.inputs()) {
      recurse(input);
    }

    if (a.has_primitive()) {
      if (auto& p = a.primitive(); typeid(p) == typeid(StopGradient)) {
        return;
      }
    }

    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        calc_grad.insert(a.id());
        for (auto& s : a.siblings()) {
          calc_grad.insert(s.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
    recurse(out);
  }

  std::unordered_map<std::uintptr_t, array> cotan_map;
  for (auto [out_idx, cotan_idx] : output_cotan_pairs) {
    auto& o = outputs[out_idx];
    auto s = o.has_primitive() ? o.primitive().stream()
                               : default_stream(default_device());
    cotan_map.insert({o.id(), astype(cotans[cotan_idx], o.dtype(), s)});
  }
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    auto& a = *it;

    std::vector<int> argnums;
    for (int i = 0; i < a.inputs().size(); ++i) {
      if (calc_grad.find(a.inputs()[i].id()) != calc_grad.end()) {
        argnums.push_back(i);
      }
    }

    auto outputs = a.outputs();
    bool has_cotans =
        std::any_of(outputs.cbegin(), outputs.cend(), [&cotan_map](auto& s) {
          return cotan_map.find(s.id()) != cotan_map.end();
        });
    if (!has_cotans) {
      continue;
    }

    auto s = a.primitive().stream();
    std::vector<array> cotangents{};
    for (auto& o : outputs) {
      if (auto cotan_it = cotan_map.find(o.id()); cotan_it != cotan_map.end()) {
        cotangents.push_back(cotan_map.extract(cotan_it).mapped());
      } else {
        cotangents.push_back(zeros_like(o, s));
      }
    }

    std::vector<array> vjps;
    {
      detail::RetainGraph retain;
      vjps = a.primitive().vjp(a.inputs(), cotangents, argnums, outputs);
    }
    for (int i = 0; i < argnums.size(); ++i) {
      auto in_id = a.inputs()[argnums[i]].id();
      if (auto cotan_it = cotan_map.find(in_id); cotan_it != cotan_map.end()) {
        cotan_it->second = add(cotan_it->second, vjps[i], s);
      } else {
        cotan_map.insert({in_id, vjps[i]});
      }
    }
  }

  std::vector<array> vjps;
  for (auto& primal : primals_) {
    if (auto cotan_it = cotan_map.find(primal.id());
        cotan_it != cotan_map.end()) {
      vjps.push_back(cotan_it->second);
    } else {
      auto s = primal.has_primitive() ? primal.primitive().stream()
                                      : default_stream(default_device());
      vjps.push_back(zeros_like(primal, s));
    }
  }
  return {outputs, vjps};
}

std::pair<array, array> vjp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& cotan) {
  auto vec_fun = [fun](const std::vector<array>& inputs) {
    return std::vector<array>{fun(inputs[0])};
  };
  auto [outputs, vjps] = vjp(vec_fun, {primal}, {cotan});
  return {outputs[0], vjps[0]};
}

std::pair<std::vector<array>, std::vector<array>> jvp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& tangents) {
  detail::InTracing in_tracing;

  if (primals.size() != tangents.size()) {
    throw std::invalid_argument("[jvp] Number of inputs does not match number of tangents.");
  }
  for (int i = 0; i < primals.size(); ++i) {
    if (primals[i].shape() != tangents[i].shape()) {
      throw std::invalid_argument("[jvp] Input shape does not match shape of tangent.");
    }
  }

  std::vector<array> primals_;
  for (auto& p : primals) {
    auto s = p.has_primitive() ? p.primitive().stream()
                               : default_stream(default_device());
    primals_.push_back(copy(p, s));
    primals_.back().set_tracer(true);
  }
  auto outputs = fun(primals_);

  std::unordered_set<std::uintptr_t> cache;
  std::unordered_set<std::uintptr_t> calc_grad;
  for (auto& primal : primals_) {
    primal.set_tracer(false);
    calc_grad.insert(primal.id());
    cache.insert(primal.id());
  }

  std::vector<array> tape;

  std::function<void(array&)> recurse;
  recurse = [&](auto& a) {
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }
    a.set_tracer(false);
    for (auto& s : a.siblings()) {
      s.set_tracer(false);
      cache.insert(s.id());
    }

    for (auto input : a.inputs()) {
      recurse(input);
    }

    if (a.has_primitive()) {
      if (auto& p = a.primitive(); typeid(p) == typeid(StopGradient)) {
        return;
      }
    }

    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        calc_grad.insert(a.id());
        for (auto& s : a.siblings()) {
          calc_grad.insert(s.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
    recurse(out);
  }

  std::unordered_map<std::uintptr_t, array> tan_map;
  for (int i = 0; i < primals_.size(); ++i) {
    tan_map.insert({primals_[i].id(), tangents[i]});
  }

  for (auto& a : tape) {
    std::vector<int> argnums;
    std::vector<array> tangents;
    for (int i = 0; i < a.inputs().size(); ++i) {
      if (auto it = tan_map.find(a.inputs()[i].id()); it != tan_map.end()) {
        argnums.push_back(i);
        tangents.push_back(it->second);
      }
    }

    auto jvps = a.primitive().jvp(a.inputs(), tangents, argnums);
    auto outputs = a.outputs();
    for (int i = 0; i < jvps.size(); ++i) {
      tan_map.insert({outputs[i].id(), jvps[i]});
    }
  }

  std::vector<array> jvps;
  for (auto& out : outputs) {
    if (auto it = tan_map.find(out.id()); it != tan_map.end()) {
      jvps.push_back(it->second);
    } else {
      auto s = out.has_primitive() ? out.primitive().stream()
                                   : default_stream(default_device());
      jvps.push_back(zeros_like(out, s));
    }
  }
  return {outputs, jvps};
}

std::pair<array, array> jvp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& tangent) {
  auto vec_fun = [fun](const std::vector<array>& inputs) {
    return std::vector<array>{fun(inputs[0])};
  };
  auto [outputs, jvps] = jvp(vec_fun, {primal}, {tangent});
  return {outputs[0], jvps[0]};
}

ValueAndGradFn value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  if (argnums.empty()) {
    throw std::invalid_argument("[grad] Must specify at least one argument.");
  }
  return [fun, argnums](const std::vector<array>& inputs) {
    std::set<int> args;
    for (auto& arg : argnums) {
      args.insert(arg < 0 ? arg + inputs.size() : arg);
    }
    if (args.size() != argnums.size()) {
      throw std::invalid_argument("[grad] Repeat argument number not allowed in grad.");
    }
    if (*args.begin() < 0 || *args.rbegin() >= inputs.size()) {
      std::ostringstream msg;
      msg << "[grad] Invalid argument number for function with "
          << inputs.size() << " inputs.";
      throw std::invalid_argument(msg.str());
    }

    auto gfun = [&fun, &inputs, &args](const std::vector<array>& ginputs) {
      std::vector<array> inputs_(inputs);
      auto argit = args.begin();
      for (int i = 0; i < ginputs.size(); ++i) {
        inputs_[*argit] = ginputs[i];
        ++argit;
      }
      auto outputs = fun(inputs_);
      for (int i = 1; i < outputs.size(); i++) {
        auto& out = outputs[i];
        auto s = out.has_primitive() ? out.primitive().stream()
                                     : default_stream(default_device());
        outputs[i] = stop_gradient(out, s);
      }
      return outputs;
    };

    std::vector<array> ginputs;
    for (auto arg : args) {
      ginputs.push_back(inputs[arg]);
    }
    auto [outputs, grads] = vjp(gfun, ginputs, {array(1.0f)});
    return std::make_pair(outputs, grads);
  };
}

}
