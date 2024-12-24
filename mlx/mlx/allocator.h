#pragma once

#include <cstdlib>

namespace mlx::core::allocator {

class Buffer {
 private:
  void* ptr_;

 public:
  Buffer(void* ptr) : ptr_(ptr) {};

  void* raw_ptr();

  const void* ptr() const {
    return ptr_;
  };
  void* ptr() {
    return ptr_;
  };
};

Buffer malloc(size_t size);

void free(Buffer buffer);

Buffer malloc_or_wait(size_t size);

class Allocator {
  /** Abstract base class for a memory allocator. */
 public:
  virtual Buffer malloc(size_t size, bool allow_swap = false) = 0;
  virtual void free(Buffer buffer) = 0;
  virtual size_t size(Buffer buffer) const = 0;

  Allocator() = default;
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) = delete;
  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) = delete;
  virtual ~Allocator() = default;
};

Allocator& allocator();

class CommonAllocator : public Allocator {
  /** A general CPU allocator. */
 public:
  virtual Buffer malloc(size_t size, bool allow_swap = false) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;

 private:
  CommonAllocator() = default;
  friend Allocator& allocator();
};

}
