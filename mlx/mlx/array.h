#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/dtype.h"
#include "mlx/event.h"

namespace mlx::core {

class Primitive;
using deleter_t = std::function<void(allocator::Buffer)>;

class array {
 public:
  /** Construct a scalar array with zero dimensions. */
  template <typename T>
  explicit array(T val, Dtype dtype = TypeToDtype<T>());

  template <typename It>
  array(
      It data,
      std::vector<int> shape,
      Dtype dtype = TypeToDtype<typename std::iterator_traits<It>::value_type>());

  template <typename T>
  array(std::initializer_list<T> data, Dtype dtype = TypeToDtype<T>());

  /* Special case so empty lists default to float32. */
  array(std::initializer_list<float> data);

  /* Special case so array({}, type) is an empty array. */
  array(std::initializer_list<int> data, Dtype dtype);

  template <typename T>
  array(
      std::initializer_list<T> data,
      std::vector<int> shape,
      Dtype dtype = TypeToDtype<T>());

  /* Build an array from a buffer */
  array(
      allocator::Buffer data,
      std::vector<int> shape,
      Dtype dtype,
      deleter_t deleter = allocator::free);

  /** Assignment to rvalue does not compile. */
  array& operator=(const array& other) && = delete;
  array& operator=(array&& other) && = delete;

  /** Default copy and move constructors otherwise. */
  array& operator=(array&& other) & = default;
  array(const array& other) = default;
  array(array&& other) = default;

  array& operator=(const array& other) & {
    if (this->id() != other.id()) {
      this->array_desc_ = other.array_desc_;
    }
    return *this;
  }

  /** The size of the array's datatype in bytes. */
  size_t itemsize() const {
    return size_of(dtype());
  }

  /** The number of elements in the array. */
  size_t size() const {
    return array_desc_->size;
  }

  /** The number of bytes in the array. */
  size_t nbytes() const {
    return size() * itemsize();
  }

  /** The number of dimensions of the array. */
  size_t ndim() const {
    return array_desc_->shape.size();
  }

  /** The shape of the array as a vector of integers. */
  const std::vector<int>& shape() const {
    return array_desc_->shape;
  }

  int shape(int dim) const {
    return shape().at(dim < 0 ? dim + ndim() : dim);
  }

  /** The strides of the array. */
  const std::vector<size_t>& strides() const {
    return array_desc_->strides;
  }

  size_t strides(int dim) const {
    return strides().at(dim < 0 ? dim + ndim() : dim);
  }

  /** Get the arrays data type. */
  Dtype dtype() const {
    return array_desc_->dtype;
  }

  /** Evaluate the array. */
  void eval();

  /** Get the value from a scalar array. */
  template <typename T>
  T item();

  template <typename T>
  T item() const;

  struct ArrayIterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = size_t;
    using value_type = const array;
    using reference = value_type;

    explicit ArrayIterator(const array& arr, int idx = 0);

    reference operator*() const;

    ArrayIterator& operator+(difference_type diff) {
      idx += diff;
      return *this;
    }

    ArrayIterator& operator++() {
      idx++;
      return *this;
    }

    friend bool operator==(const ArrayIterator& a, const ArrayIterator& b) {
      return a.arr.id() == b.arr.id() && a.idx == b.idx;
    }
    friend bool operator!=(const ArrayIterator& a, const ArrayIterator& b) {
      return !(a == b);
    }

   private:
    const array& arr;
    int idx;
  };

  ArrayIterator begin() const {
    return ArrayIterator(*this);
  }
  ArrayIterator end() const {
    return ArrayIterator(*this, shape(0));
  }

  array(
      std::vector<int> shape,
      Dtype dtype,
      std::shared_ptr<Primitive> primitive,
      std::vector<array> inputs);

  static std::vector<array> make_arrays(
      std::vector<std::vector<int>> shapes,
      const std::vector<Dtype>& dtypes,
      const std::shared_ptr<Primitive>& primitive,
      const std::vector<array>& inputs);

  /** A unique identifier for an array. */
  std::uintptr_t id() const {
    return reinterpret_cast<std::uintptr_t>(array_desc_.get());
  }

  /** A unique identifier for an arrays primitive. */
  std::uintptr_t primitive_id() const {
    return reinterpret_cast<std::uintptr_t>(array_desc_->primitive.get());
  }

  struct Data {
    allocator::Buffer buffer;
    deleter_t d;
    Data(allocator::Buffer buffer, deleter_t d = allocator::free)
        : buffer(buffer), d(d) {}
    Data(const Data& d) = delete;
    Data& operator=(const Data& d) = delete;
    ~Data() {
      d(buffer);
    }
  };

  struct Flags {
    bool contiguous : 1;

    bool row_contiguous : 1;

    bool col_contiguous : 1;
  };

  /** The array's primitive. */
  Primitive& primitive() const {
    return *(array_desc_->primitive);
  }

  /** A shared pointer to the array's primitive. */
  std::shared_ptr<Primitive>& primitive_ptr() const {
    return array_desc_->primitive;
  }

  /** Check if the array has an attached primitive or is a leaf node. */
  bool has_primitive() const {
    return array_desc_->primitive != nullptr;
  }

  /** The array's inputs. */
  const std::vector<array>& inputs() const {
    return array_desc_->inputs;
  }

  std::vector<array>& inputs() {
    return array_desc_->inputs;
  }

  /** True indicates the arrays buffer is safe to reuse */
  bool is_donatable() const {
    return array_desc_.use_count() == 1 && (array_desc_->data.use_count() == 1);
  }

  /** The array's siblings. */
  const std::vector<array>& siblings() const {
    return array_desc_->siblings;
  }

  /** The array's siblings. */
  std::vector<array>& siblings() {
    return array_desc_->siblings;
  }

  void set_siblings(std::vector<array> siblings, uint16_t position) {
    array_desc_->siblings = std::move(siblings);
    array_desc_->position = position;
  }

  /** The outputs of the array's primitive (i.e. this array and
   * its siblings) in the order the primitive expects. */
  std::vector<array> outputs() const {
    auto idx = array_desc_->position;
    std::vector<array> outputs;
    outputs.reserve(siblings().size() + 1);
    outputs.insert(outputs.end(), siblings().begin(), siblings().begin() + idx);
    outputs.push_back(*this);
    outputs.insert(outputs.end(), siblings().begin() + idx, siblings().end());
    return outputs;
  }

  /** Detach the array from the graph. */
  void detach();

  /** Get the Flags bit-field. */
  const Flags& flags() const {
    return array_desc_->flags;
  }

  /** The size (in elements) of the underlying buffer the array points to.
   *
   * This can be different than the actual size of the array if the array has
   * been broadcast or irregularly strided.  If ``first`` is the offset into
   * the data buffer of the first element of the array (i.e. the offset
   * corresponding to ``arr[0, 0, ...]``) and last is the offset into the
   * data buffer of the last element of the array (i.e. the offset
   * corresponding to ``arr[-1, -1, ...]``) then ``data_size = last - first``.
   * Note, ``data_size`` is in units of ``item_size`` (not bytes).
   **/
  size_t data_size() const {
    return array_desc_->data_size;
  }

  allocator::Buffer& buffer() {
    return array_desc_->data->buffer;
  }
  const allocator::Buffer& buffer() const {
    return array_desc_->data->buffer;
  }

  size_t buffer_size() const {
    return allocator::allocator().size(buffer());
  }

  std::shared_ptr<Data> data_shared_ptr() const {
    return array_desc_->data;
  }
  template <typename T>
  T* data() {
    return static_cast<T*>(array_desc_->data_ptr);
  }

  template <typename T>
  const T* data() const {
    return static_cast<T*>(array_desc_->data_ptr);
  }

  enum Status { unscheduled, scheduled, available };

  bool is_available() const {
    return status() == Status::available;
  }

  Status status() const {
    return array_desc_->status;
  }

  void set_status(Status s) const {
    array_desc_->status = s;
  }

  Event& event() const {
    return array_desc_->event;
  }

  void attach_event(Event e) const {
    array_desc_->event = std::move(e);
  }

  void set_tracer(bool is_tracer) {
    array_desc_->is_tracer = is_tracer;
  }
  bool is_tracer() const;

  void set_data(allocator::Buffer buffer, deleter_t d = allocator::free);

  void set_data(
      allocator::Buffer buffer,
      size_t data_size,
      std::vector<size_t> strides,
      Flags flags,
      deleter_t d = allocator::free);

  void copy_shared_buffer(
      const array& other,
      const std::vector<size_t>& strides,
      Flags flags,
      size_t data_size,
      size_t offset = 0);

  void copy_shared_buffer(const array& other);

  void move_shared_buffer(
      array other,
      const std::vector<size_t>& strides,
      Flags flags,
      size_t data_size,
      size_t offset = 0);

  void move_shared_buffer(array other);

  void overwrite_descriptor(const array& other) {
    array_desc_ = other.array_desc_;
  }

  ~array();

 private:
  template <typename It>
  void init(const It src);

  struct ArrayDesc {
    std::vector<int> shape;
    std::vector<size_t> strides;
    size_t size;
    Dtype dtype;
    std::shared_ptr<Primitive> primitive;

    Status status;

    Event event;

    bool is_tracer{false};

    std::shared_ptr<Data> data;

    void* data_ptr{nullptr};

    size_t data_size;

    Flags flags;

    std::vector<array> inputs;
    std::vector<array> siblings;
    uint32_t position{0};

    explicit ArrayDesc(std::vector<int> shape, Dtype dtype);

    explicit ArrayDesc(
        std::vector<int> shape,
        Dtype dtype,
        std::shared_ptr<Primitive> primitive,
        std::vector<array> inputs);

    ~ArrayDesc();

   private:
    void init();
  };

  std::shared_ptr<ArrayDesc> array_desc_;
};

template <typename T>
array::array(T val, Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(std::vector<int>{}, dtype)) {
  init(&val);
}

template <typename It>
array::array(
  It data,
  std::vector<int> shape,
  Dtype dtype /* = TypeToDtype<typename std::iterator_traits<It>::value_type>() */) :
    array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  init(data);
}

template <typename T>
array::array(
    std::initializer_list<T> data,
    Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(
          std::vector<int>{static_cast<int>(data.size())},
          dtype)) {
  init(data.begin());
}

template <typename T>
array::array(
    std::initializer_list<T> data,
    std::vector<int> shape,
    Dtype dtype /* = TypeToDtype<T>() */)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  if (data.size() != size()) {
    throw std::invalid_argument("Data size and provided shape mismatch in array construction.");
  }
  init(data.begin());
}

template <typename T>
T array::item() {
  if (size() != 1) {
    throw std::invalid_argument("item can only be called on arrays of size 1.");
  }
  eval();
  return *data<T>();
}

template <typename T>
T array::item() const {
  if (size() != 1) {
    throw std::invalid_argument("item can only be called on arrays of size 1.");
  }
  if (status() == Status::unscheduled) {
    throw std::invalid_argument("item() const can only be called on evaled arrays");
  }
  const_cast<array*>(this)->eval();
  return *data<T>();
}

template <typename It>
void array::init(It src) {
  set_data(allocator::malloc(size() * size_of(dtype())));
  switch (dtype()) {
    case bool_:
      std::copy(src, src + size(), data<bool>());
      break;
    case uint8:
      std::copy(src, src + size(), data<uint8_t>());
      break;
    case uint16:
      std::copy(src, src + size(), data<uint16_t>());
      break;
    case uint32:
      std::copy(src, src + size(), data<uint32_t>());
      break;
    case uint64:
      std::copy(src, src + size(), data<uint64_t>());
      break;
    case int8:
      std::copy(src, src + size(), data<int8_t>());
      break;
    case int16:
      std::copy(src, src + size(), data<int16_t>());
      break;
    case int32:
      std::copy(src, src + size(), data<int32_t>());
      break;
    case int64:
      std::copy(src, src + size(), data<int64_t>());
      break;
    case float16:
      std::copy(src, src + size(), data<float16_t>());
      break;
    case float32:
      std::copy(src, src + size(), data<float>());
      break;
    case bfloat16:
      std::copy(src, src + size(), data<bfloat16_t>());
      break;
  }
}

template <typename T>
inline constexpr bool is_array_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, array>;

template <typename... T>
inline constexpr bool is_arrays_v = (is_array_v<T> && ...);

template <typename... T>
using enable_for_arrays_t = typename std::enable_if_t<is_arrays_v<T...>>;

}
