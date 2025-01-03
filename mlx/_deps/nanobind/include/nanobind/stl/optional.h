#pragma once

#include "detail/nb_optional.h"
#include <optional>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T> struct remove_opt_mono<std::optional<T>>
    : remove_opt_mono<T> { };

template <typename T>
struct type_caster<std::optional<T>> : optional_caster<std::optional<T>> {};

template <> struct type_caster<std::nullopt_t> : none_caster<std::nullopt_t> { };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
