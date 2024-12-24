#include "mlx/backend/common/compiled.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

std::string get_type_string(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "bfloat16_t";
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    default: {
      std::ostringstream msg;
      msg << "Unsupported compilation type " << d;
      throw std::runtime_error(msg.str());
    }
  }
}

}
