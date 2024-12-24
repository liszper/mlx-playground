#pragma once

#include <variant>

#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {

void save(std::shared_ptr<io::Writer> out_stream, array a);

void save(std::string file, array a);

array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

array load(std::string file, StreamOrDevice s = {});

}
