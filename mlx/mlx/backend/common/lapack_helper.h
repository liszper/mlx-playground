#pragma once

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

#if defined(LAPACK_GLOBAL) || defined(LAPACK_NAME)

#define MLX_LAPACK_FUNC(f) LAPACK_##f

#else

#define MLX_LAPACK_FUNC(f) f##_

#endif
