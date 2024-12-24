#pragma once

#include <memory>
#include <stdexcept>

#include "mlx/stream.h"

namespace mlx::core {

class Event {
 public:
  Event() = default;

  Event(const Stream& steam);

  void wait();

  void signal();

  bool valid() const {
    return event_ != nullptr;
  }

  uint64_t value() const {
    return value_;
  }

  void set_value(uint64_t v) {
    value_ = v;
  }

  const Stream& stream() const {
    if (!valid()) {
      throw std::runtime_error("[Event::stream] Cannot access stream on invalid event.");
    }
    return stream_;
  }

  const std::shared_ptr<void>& raw_event() const {
    return event_;
  }

 private:
  Stream stream_{0, Device::cpu};
  std::shared_ptr<void> event_;
  uint64_t value_{0};
};

}
