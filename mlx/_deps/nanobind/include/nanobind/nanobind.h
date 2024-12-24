#pragma once

#if __cplusplus < 201703L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)
#  error The nanobind library requires C++17!
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4702)
#  pragma warning(disable: 4275)
#  pragma warning(disable: 4251)
#endif

#define NB_VERSION_MAJOR 2
#define NB_VERSION_MINOR 2
#define NB_VERSION_PATCH 0
#define NB_VERSION_DEV   0

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <new>

#include "nb_python.h"
#include "nb_defs.h"
#include "nb_enums.h"
#include "nb_traits.h"
#include "nb_tuple.h"
#include "nb_lib.h"
#include "nb_descr.h"
#include "nb_types.h"
#include "nb_accessor.h"
#include "nb_error.h"
#include "nb_attr.h"
#include "nb_cast.h"
#include "nb_misc.h"
#include "nb_call.h"
#include "nb_func.h"
#include "nb_class.h"

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
