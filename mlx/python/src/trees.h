#pragma once
#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace nb = nanobind;
using namespace mlx::core;

void tree_visit(
    const std::vector<nb::object>& trees,
    std::function<void(const std::vector<nb::object>&)> visitor);
void tree_visit(nb::object tree, std::function<void(nb::handle)> visitor);

nb::object tree_map(
    const std::vector<nb::object>& trees,
    std::function<nb::object(const std::vector<nb::object>&)> transform);

nb::object tree_map(
    nb::object tree,
    std::function<nb::object(nb::handle)> transform);

void tree_visit_update(
    nb::object tree,
    std::function<nb::object(nb::handle)> visitor);

void tree_fill(nb::object& tree, const std::vector<array>& values);

void tree_replace(
    nb::object& tree,
    const std::vector<array>& src,
    const std::vector<array>& dst);

std::vector<array> tree_flatten(nb::object tree, bool strict = true);

nb::object tree_unflatten(
    nb::object tree,
    const std::vector<array>& values,
    int index = 0);

std::pair<std::vector<array>, nb::object> tree_flatten_with_structure(
    nb::object tree,
    bool strict = true);

nb::object tree_unflatten_from_structure(
    nb::object structure,
    const std::vector<array>& values,
    int index = 0);
