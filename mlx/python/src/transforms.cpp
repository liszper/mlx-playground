#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

#include "mlx/array.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"
#include "python/src/trees.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

using IntOrVec = std::variant<int, std::vector<int>>;
using StrOrVec = std::variant<std::string, std::vector<std::string>>;

inline std::string type_name_str(const nb::handle& o) {
  return nb::cast<std::string>(nb::type_name(o.type()));
}

template <typename T>
std::vector<T> to_vector(const std::variant<T, std::vector<T>>& v) {
  std::vector<T> vals;
  if (auto pv = std::get_if<T>(&v); pv) {
    vals.push_back(*pv);
  } else {
    vals = std::get<std::vector<T>>(v);
  }
  return vals;
}

auto validate_argnums_argnames(
    const std::optional<IntOrVec>& argnums, const StrOrVec& argnames) {
  auto vec_names = to_vector(argnames);

  if (!argnums.has_value()) {
    if (vec_names.empty()) {
      return std::make_pair(std::vector<int>{0}, vec_names);
    } else {
      return std::make_pair(std::vector<int>{}, vec_names);
    }
  }

  return std::make_pair(to_vector(*argnums), vec_names);
}

auto py_value_and_grad(
    const nb::callable& fun, std::vector<int> argnums, std::vector<std::string> argnames, const std::string& error_msg_tag, bool scalar_func_only) {
  if (argnums.size() == 0 && argnames.size() == 0) {
    throw std::invalid_argument(
        error_msg_tag + " Gradient wrt no argument requested");
  }
  if (argnums.size() > 0) {
    std::sort(argnums.begin(), argnums.end());
    if (argnums[0] < 0) {
      std::ostringstream msg;
      msg << error_msg_tag
          << " Can't compute the gradient of negative argument index "
          << argnums[0];
      throw std::invalid_argument(msg.str());
    }
  }

  return [fun, argnums, argnames, error_msg_tag, scalar_func_only](
             const nb::args& args, const nb::kwargs& kwargs) {
    if (argnums.size() > 0 && argnums.back() >= args.size()) {
      std::ostringstream msg;
      msg << error_msg_tag << " Can't compute the gradient of argument index "
          << argnums.back() << " because the function is called with only "
          << args.size() << " positional arguments.";
      throw std::invalid_argument(msg.str());
    }

    for (auto& key : argnames) {
      if (!kwargs.contains(key)) {
        std::ostringstream msg;
        msg << error_msg_tag
            << " Can't compute the gradient of keyword argument '" << key
            << "' because the function is called with the "
            << "following keyword arguments {";
        for (auto item : kwargs) {
          msg << nb::cast<std::string>(item.first) << ",";
        }
        msg << "}";
        throw std::invalid_argument(msg.str());
      }
    }

    std::vector<array> arrays;
    std::vector<int> counts(1, 0);
    for (auto i : argnums) {
      auto argsi = tree_flatten(args[i]);
      arrays.insert(arrays.end(), argsi.begin(), argsi.end());
      counts.push_back(argsi.size());
    }
    for (auto& key : argnames) {
      auto argsk = tree_flatten(kwargs[key.c_str()]);
      arrays.insert(arrays.end(), argsk.begin(), argsk.end());
      counts.push_back(argsk.size());
    }
    std::partial_sum(counts.cbegin(), counts.cend(), counts.begin());
    std::vector<int> gradient_indices(arrays.size());
    std::iota(gradient_indices.begin(), gradient_indices.end(), 0);

    nb::object py_value_out;
    auto value_and_grads = value_and_grad(
        [&fun, &args, &kwargs, &argnums, &argnames, &counts, &py_value_out, &error_msg_tag, scalar_func_only](const std::vector<array>& a) {
          nb::list args_cpy;
          nb::kwargs kwargs_cpy = nb::kwargs();
          int j = 0;
          for (int i = 0; i < args.size(); ++i) {
            if (j < argnums.size() && i == argnums[j]) {
              args_cpy.append(tree_unflatten(args[i], a, counts[j]));
              j++;
            } else {
              args_cpy.append(args[i]);
            }
          }
          for (auto& key : argnames) {
            kwargs_cpy[key.c_str()] = tree_unflatten(kwargs[key.c_str()], a, counts[j]);
            j++;
          }
          for (auto item : kwargs) {
            if (kwargs_cpy.contains(item.first)) {
              continue;
            }
            kwargs_cpy[item.first] = item.second;
          }

          py_value_out = fun(*args_cpy, **kwargs_cpy);

          if (!nb::isinstance<array>(py_value_out)) {
            if (scalar_func_only) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be a "
                  << "scalar array; but " << type_name_str(py_value_out)
                  << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<nb::tuple>(py_value_out)) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, tuple[array, Any, ...]]); but "
                  << type_name_str(py_value_out) << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            nb::tuple ret = nb::cast<nb::tuple>(py_value_out);
            if (ret.size() == 0) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a non-empty tuple. The first value should be a "
                  << "scalar array and the rest can be anything. Instead, "
                  << "we got an empty tuple.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<array>(ret[0])) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, tuple[array, Any, ...]]); but it "
                  << "was a tuple with the first value being of type "
                  << type_name_str(ret[0]) << " .";
              throw std::invalid_argument(msg.str());
            }
          }

          return tree_flatten(py_value_out, false);
        },
        gradient_indices)(arrays);

    auto value = value_and_grads.first;
    auto gradients = value_and_grads.second;

    nb::object positional_grads;
    nb::object keyword_grads;
    nb::object py_grads;

    if (argnums.size() == 1) {
      positional_grads = tree_unflatten(args[argnums[0]], gradients, counts[0]);
    } else if (argnums.size() > 1) {
      nb::list grads_;
      for (int i = 0; i < argnums.size(); i++) {
        grads_.append(tree_unflatten(args[argnums[i]], gradients, counts[i]));
      }
      positional_grads = nb::tuple(grads_);
    } else {
      positional_grads = nb::none();
    }

    if (argnames.size() == 0) {
      py_grads = positional_grads;
    } else {
      nb::dict grads_;
      for (int i = 0; i < argnames.size(); i++) {
        auto& k = argnames[i];
        grads_[k.c_str()] = tree_unflatten(
            kwargs[k.c_str()], gradients, counts[i + argnums.size()]);
      }
      keyword_grads = grads_;

      py_grads = nb::make_tuple(positional_grads, keyword_grads);
    }

    nb::object return_value = tree_unflatten(py_value_out, value);
    return std::make_pair(return_value, py_grads);
  };
}

void init_transforms(nb::module_& m) {
  m.def("eval", [](const nb::args& args) {
        std::vector<array> arrays = tree_flatten(args, false);
        {
          nb::gil_scoped_release nogil;
          eval(arrays);
        }
      },
      nb::arg(),
      nb::sig("def eval(*args) -> None"));
  m.def("async_eval", [](const nb::args& args) {
        std::vector<array> arrays = tree_flatten(args, false);
        {
          nb::gil_scoped_release nogil;
          async_eval(arrays);
        }
      },
      nb::arg(),
      nb::sig("def async_eval(*args)"));
  m.def("value_and_grad", [](const nb::callable& fun, const std::optional<IntOrVec>& argnums, const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] = validate_argnums_argnames(argnums, argnames);
        return nb::cpp_function(py_value_and_grad(
            fun, argnums_vec, argnames_vec, "[value_and_grad]", false));
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig("def value_and_grad(fun: Callable, argnums: Optional[Union[int, list[int]]] = None, argnames: Union[str, list[str]] = []) -> Callable"));
  m.def("grad", [](const nb::callable& fun, const std::optional<IntOrVec>& argnums, const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] = validate_argnums_argnames(argnums, argnames);
        auto fn = py_value_and_grad(fun, argnums_vec, argnames_vec, "[grad]", true);
        return nb::cpp_function(
            [fn](const nb::args& args, const nb::kwargs& kwargs) {
              return fn(args, kwargs).second;
            });
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig("def grad(fun: Callable, argnums: Optional[Union[int, list[int]]] = None, argnames: Union[str, list[str]] = []) -> Callable"));
}
