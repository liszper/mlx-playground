#pragma once

#include "mlx/device.h"

namespace mlx::core {

struct Stream {
  int index;
  Device device;
  explicit Stream(int index, Device device) : index(index), device(device) {}
};

Stream default_stream(Device d);

void set_default_stream(Stream s);

Stream new_stream(Device d);

inline bool operator==(const Stream& lhs, const Stream& rhs) {
  return lhs.index == rhs.index;
}

inline bool operator!=(const Stream& lhs, const Stream& rhs) {
  return !(lhs == rhs);
}

void synchronize();

void synchronize(Stream);

}
