#pragma once

#include <unordered_set>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"

#define DEFINE_GRADS()                           \
  std::vector<array> jvp(                        \
      const std::vector<array>& primals,         \
      const std::vector<array>& tangents,        \
      const std::vector<int>& argnums) override; \
                                                 \
  std::vector<array> vjp(                        \
      const std::vector<array>& primals,         \
      const std::vector<array>& cotangents,      \
      const std::vector<int>& argnums,           \
      const std::vector<array>& outputs) override;

#define DEFINE_PRINT(PRIMITIVE)           \
  void print(std::ostream& os) override { \
    os << #PRIMITIVE;                     \
  }

#define DEFINE_DEFAULT_IS_EQUIVALENT()                        \
  bool is_equivalent(const Primitive& other) const override { \
    return true;                                              \
  }

#define DEFINE_INPUT_OUTPUT_SHAPE()                \
  std::vector<std::vector<int>> output_shapes(     \
      const std::vector<array>& inputs) override { \
    return {inputs[0].shape()};                    \
  }

namespace mlx::core {

class Primitive {
 public:
  explicit Primitive(Stream stream) : stream_(stream) {}

  /** The device the primitive will run on. */
  const Device& device() {
    return stream().device;
  }

  /** The stream the primitive will run on. */
  const Stream& stream() {
    return stream_;
  }

  virtual void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;
  virtual void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;

  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums);

  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs);

  /** Print the primitive. */
  virtual void print(std::ostream& os) = 0;

  /** Equivalence check defaults to false unless overridden by the primitive */
  virtual bool is_equivalent(const Primitive& other) const {
    return false;
  }

  /** Get the output shapes of the primitive. This is not required to be
   * implemented by derived classes, in which case it will throw. */
  virtual std::vector<std::vector<int>> output_shapes(
      const std::vector<array>& inputs);

  virtual ~Primitive() = default;
  Primitive(const Primitive& other) = delete;
  Primitive(Primitive&& other) = delete;
  Primitive& operator=(const Primitive& other) = delete;
  Primitive& operator=(Primitive&& other) = delete;

 private:
  Stream stream_;
};

class UnaryPrimitive : public Primitive {
 public:
  explicit UnaryPrimitive(Stream stream) : Primitive(stream) {}

  virtual void eval_cpu(const std::vector<array>& inputs, array& output) = 0;
  virtual void eval_gpu(const std::vector<array>& inputs, array& output) = 0;

  inline void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    eval_cpu(inputs, outputs[0]);
  }
  inline void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    eval_gpu(inputs, outputs[0]);
  }

  virtual ~UnaryPrimitive() = default;
  UnaryPrimitive(const UnaryPrimitive& other) = delete;
  UnaryPrimitive(UnaryPrimitive&& other) = delete;
  UnaryPrimitive& operator=(const UnaryPrimitive& other) = delete;
  UnaryPrimitive& operator=(UnaryPrimitive&& other) = delete;
};

class Abs : public UnaryPrimitive {
 public:
  explicit Abs(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Abs)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Add : public UnaryPrimitive {
 public:
  explicit Add(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Add)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Arange : public UnaryPrimitive {
 public:
  explicit Arange(Stream stream, double start, double stop, double step)
      : UnaryPrimitive(stream), start_(start), stop_(stop), step_(step) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Arange)
  bool is_equivalent(const Primitive& other) const override;

 private:
  double start_;
  double stop_;
  double step_;

  void eval(const std::vector<array>& inputs, array& out);
};

class ArgReduce : public UnaryPrimitive {
 public:
  enum ReduceType {
    ArgMin,
    ArgMax,
  };

  explicit ArgReduce(Stream stream, ReduceType reduce_type, int axis)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ArgReduce)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<std::vector<int>> output_shapes(
      const std::vector<array>& inputs) override;

 private:
  ReduceType reduce_type_;
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class AsType : public UnaryPrimitive {
 public:
  explicit AsType(Stream stream, Dtype dtype)
      : UnaryPrimitive(stream), dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(AsType)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;

 private:
  Dtype dtype_;

  void eval(const std::vector<array>& inputs, array& out);
};

class AsStrided : public UnaryPrimitive {
 public:
  explicit AsStrided(
      Stream stream,
      std::vector<int> shape,
      std::vector<size_t> strides,
      size_t offset)
      : UnaryPrimitive(stream),
        shape_(std::move(shape)),
        strides_(std::move(strides)),
        offset_(offset) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(AsStrided)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;
  std::vector<size_t> strides_;
  size_t offset_;

  void eval(const std::vector<array>& inputs, array& out);
};

class BitwiseBinary : public UnaryPrimitive {
 public:
  enum Op { And, Or, Xor, LeftShift, RightShift };

  explicit BitwiseBinary(Stream stream, Op op)
      : UnaryPrimitive(stream), op_(op) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  bool is_equivalent(const Primitive& other) const override;
  void print(std::ostream& os) override;
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  Op op_;
};

class Broadcast : public UnaryPrimitive {
 public:
  explicit Broadcast(Stream stream, const std::vector<int>& shape)
      : UnaryPrimitive(stream), shape_(shape) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Broadcast)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Ceil : public UnaryPrimitive {
 public:
  explicit Ceil(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Ceil)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Concatenate : public UnaryPrimitive {
 public:
  explicit Concatenate(Stream stream, int axis)
      : UnaryPrimitive(stream), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Concatenate)
  bool is_equivalent(const Primitive& other) const override;

 private:
  int axis_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Copy : public UnaryPrimitive {
 public:
  explicit Copy(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Copy)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Cos : public UnaryPrimitive {
 public:
  explicit Cos(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Cos)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Cosh : public UnaryPrimitive {
 public:
  explicit Cosh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Cosh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Depends : public Primitive {
 public:
  explicit Depends(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotan,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(Depends);

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

class Divide : public UnaryPrimitive {
 public:
  explicit Divide(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Divide)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class DivMod : public Primitive {
 public:
  explicit DivMod(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_GRADS()
  DEFINE_PRINT(DivMod)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  std::vector<std::vector<int>> output_shapes(
      const std::vector<array>& inputs) override {
    return std::vector{inputs[0].shape(), inputs[0].shape()};
  }

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

class Select : public UnaryPrimitive {
 public:
  explicit Select(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Select)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Remainder : public UnaryPrimitive {
 public:
  explicit Remainder(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Remainder)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Equal : public UnaryPrimitive {
 public:
  explicit Equal(Stream stream, bool equal_nan = false)
      : UnaryPrimitive(stream), equal_nan_(equal_nan) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

  void print(std::ostream& os) override {
    if (equal_nan_) {
      os << "NaNEqual";
    } else {
      os << "Equal";
    }
  }

 private:
  void eval(const std::vector<array>& inputs, array& out);
  bool equal_nan_;
};

class Erf : public UnaryPrimitive {
 public:
  explicit Erf(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Erf)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class ErfInv : public UnaryPrimitive {
 public:
  explicit ErfInv(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(ErfInv)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Exp : public UnaryPrimitive {
 public:
  explicit Exp(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Exp)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Expm1 : public UnaryPrimitive {
 public:
  explicit Expm1(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Expm1)
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Floor : public UnaryPrimitive {
 public:
  explicit Floor(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Floor)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Full : public UnaryPrimitive {
 public:
  explicit Full(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Full)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Gather : public UnaryPrimitive {
 public:
  explicit Gather(
      Stream stream,
      const std::vector<int>& axes,
      const std::vector<int>& slice_sizes)
      : UnaryPrimitive(stream), axes_(axes), slice_sizes_(slice_sizes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Gather)
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  std::vector<int> axes_;
  std::vector<int> slice_sizes_;
};

class Greater : public UnaryPrimitive {
 public:
  explicit Greater(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Greater)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class GreaterEqual : public UnaryPrimitive {
 public:
  explicit GreaterEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(GreaterEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Less : public UnaryPrimitive {
 public:
  explicit Less(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Less)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LessEqual : public UnaryPrimitive {
 public:
  explicit LessEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LessEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Load : public UnaryPrimitive {
 public:
  explicit Load(
      Stream stream,
      std::shared_ptr<io::Reader> reader,
      size_t offset,
      bool swap_endianness = false)
      : UnaryPrimitive(stream),
        reader_(std::move(reader)),
        offset_(offset),
        swap_endianness_(swap_endianness) {
    if (stream.device == Device::gpu) {
      io_stream();
    }
  }

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Load)

 private:
  Stream& io_stream() {
    static Stream io_stream = new_stream(Device::cpu);
    return io_stream;
  };
  void eval(const std::vector<array>& inputs, array& out);
  std::shared_ptr<io::Reader> reader_;
  size_t offset_;
  bool swap_endianness_;
};

class Log : public UnaryPrimitive {
 public:
  enum Base { two, ten, e };

  explicit Log(Stream stream, Base base)
      : UnaryPrimitive(stream), base_(base) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

  void print(std::ostream& os) override {
    switch (base_) {
      case e:
        os << "Log";
        break;
      case two:
        os << "Log2";
        break;
      case ten:
        os << "Log10";
        break;
    }
  }

 private:
  Base base_;
  void eval(const std::vector<array>& inputs, array& out);
};

class Log1p : public UnaryPrimitive {
 public:
  explicit Log1p(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Log1p)
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LogicalNot : public UnaryPrimitive {
 public:
  explicit LogicalNot(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LogicalNot)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LogicalAnd : public UnaryPrimitive {
 public:
  explicit LogicalAnd(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LogicalAnd)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class LogicalOr : public UnaryPrimitive {
 public:
  explicit LogicalOr(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(LogicalOr)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Matmul : public UnaryPrimitive {
 public:
  explicit Matmul(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(Matmul)
  DEFINE_DEFAULT_IS_EQUIVALENT()
};

class Maximum : public UnaryPrimitive {
 public:
  explicit Maximum(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Maximum)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Minimum : public UnaryPrimitive {
 public:
  explicit Minimum(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Minimum)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Multiply : public UnaryPrimitive {
 public:
  explicit Multiply(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Multiply)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Negative : public UnaryPrimitive {
 public:
  explicit Negative(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Negative)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class NotEqual : public UnaryPrimitive {
 public:
  explicit NotEqual(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(NotEqual)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class NumberOfElements : public UnaryPrimitive {
 public:
  explicit NumberOfElements(
      Stream stream,
      std::vector<int> axes,
      bool inverted,
      Dtype dtype)
      : UnaryPrimitive(stream),
        axes_(std::move(axes)),
        inverted_(inverted),
        dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(NumberOfElements)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<std::vector<int>> output_shapes(
      const std::vector<array>& inputs) override {
    return {{}};
  }

 private:
  std::vector<int> axes_;
  bool inverted_;
  Dtype dtype_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Pad : public UnaryPrimitive {
 public:
  explicit Pad(
      Stream stream,
      const std::vector<int>& axes,
      const std::vector<int>& low_pad_size,
      const std::vector<int>& high_pad_size)
      : UnaryPrimitive(stream),
        axes_(axes),
        low_pad_size_(low_pad_size),
        high_pad_size_(high_pad_size) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Pad)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> axes_;
  std::vector<int> low_pad_size_;
  std::vector<int> high_pad_size_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Power : public UnaryPrimitive {
 public:
  explicit Power(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Power)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class RandomBits : public UnaryPrimitive {
 public:
  explicit RandomBits(Stream stream, const std::vector<int>& shape, int width)
      : UnaryPrimitive(stream), shape_(shape), width_(width) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(RandomBits)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;
  int width_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Reshape : public UnaryPrimitive {
 public:
  explicit Reshape(Stream stream, const std::vector<int>& shape)
      : UnaryPrimitive(stream), shape_(shape) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Reshape)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> shape_;

  void eval(const std::vector<array>& inputs, array& out);

  std::pair<bool, std::vector<size_t>> prepare_reshape(
      const array& in,
      const array& out);
  void shared_buffer_reshape(
      const array& in,
      const std::vector<size_t>& out_strides,
      array& out);
};

class Reduce : public UnaryPrimitive {
 public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };

  explicit Reduce(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::vector<std::vector<int>> output_shapes(
      const std::vector<array>& inputs) override;

  void print(std::ostream& os) override {
    switch (reduce_type_) {
      case And:
        os << "And";
        break;
      case Or:
        os << "Or";
        break;
      case Sum:
        os << "Sum";
        break;
      case Prod:
        os << "Prod";
        break;
      case Min:
        os << "Min";
        break;
      case Max:
        os << "Max";
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  ReduceType reduce_type_;
  std::vector<int> axes_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Round : public UnaryPrimitive {
 public:
  explicit Round(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Round)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Scan : public UnaryPrimitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod };

  explicit Scan(
      Stream stream,
      ReduceType reduce_type,
      int axis,
      bool reverse,
      bool inclusive)
      : UnaryPrimitive(stream),
        reduce_type_(reduce_type),
        axis_(axis),
        reverse_(reverse),
        inclusive_(inclusive) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS();

  void print(std::ostream& os) override {
    os << "Cum";
    switch (reduce_type_) {
      case Sum:
        os << "Sum";
        break;
      case Prod:
        os << "Prod";
        break;
      case Min:
        os << "Min";
        break;
      case Max:
        os << "Max";
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  ReduceType reduce_type_;
  int axis_;
  bool reverse_;
  bool inclusive_;

  void eval(const std::vector<array>& inputs, array& out);
};

class Scatter : public UnaryPrimitive {
 public:
  enum ReduceType { Max, Min, Sum, Prod, None };

  explicit Scatter(
      Stream stream,
      ReduceType reduce_type,
      const std::vector<int>& axes)
      : UnaryPrimitive(stream), reduce_type_(reduce_type), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS();

  void print(std::ostream& os) override {
    os << "Scatter";
    switch (reduce_type_) {
      case Sum:
        os << " Sum";
        break;
      case Prod:
        os << " Prod";
        break;
      case Min:
        os << " Min";
        break;
      case Max:
        os << " Max";
        break;
      case None:
        break;
    }
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  ReduceType reduce_type_;
  std::vector<int> axes_;
};

class Sigmoid : public UnaryPrimitive {
 public:
  explicit Sigmoid(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sigmoid)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sign : public UnaryPrimitive {
 public:
  explicit Sign(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sign)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sin : public UnaryPrimitive {
 public:
  explicit Sin(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sin)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sinh : public UnaryPrimitive {
 public:
  explicit Sinh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Sinh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Slice : public UnaryPrimitive {
 public:
  explicit Slice(
      Stream stream,
      const std::vector<int>& start_indices,
      const std::vector<int>& end_indices,
      const std::vector<int>& strides)
      : UnaryPrimitive(stream),
        start_indices_(start_indices),
        end_indices_(end_indices),
        strides_(strides) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Slice)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> start_indices_;
  std::vector<int> end_indices_;
  std::vector<int> strides_;

  void eval(const std::vector<array>& inputs, array& out);
};

class SliceUpdate : public UnaryPrimitive {
 public:
  explicit SliceUpdate(
      Stream stream,
      const std::vector<int>& start_indices,
      const std::vector<int>& end_indices,
      const std::vector<int>& strides)
      : UnaryPrimitive(stream),
        start_indices_(start_indices),
        end_indices_(end_indices),
        strides_(strides) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(SliceUpdate)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> start_indices_;
  std::vector<int> end_indices_;
  std::vector<int> strides_;

  void eval(const std::vector<array>& inputs, array& out);

  std::tuple<int64_t, std::vector<int64_t>> prepare_slice(const array& in);
};

class Softmax : public UnaryPrimitive {
 public:
  explicit Softmax(Stream stream, bool precise)
      : UnaryPrimitive(stream), precise_(precise) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Softmax)
  DEFINE_INPUT_OUTPUT_SHAPE()

  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, array& out);
  bool precise_;
};

class Split : public Primitive {
 public:
  explicit Split(Stream stream, const std::vector<int>& indices, int axis)
      : Primitive(stream), indices_(indices), axis_(axis) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_GRADS()
  DEFINE_PRINT(Split)
  bool is_equivalent(const Primitive& other) const override;

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

  std::vector<int> indices_;
  int axis_;
};

class Square : public UnaryPrimitive {
 public:
  explicit Square(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Square)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Sqrt : public UnaryPrimitive {
 public:
  explicit Sqrt(Stream stream, bool recip = false)
      : UnaryPrimitive(stream), recip_(recip) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override;

  void print(std::ostream& os) override {
    if (recip_) {
      os << "Rsqrt";
    } else {
      os << "Sqrt";
    }
  }

 private:
  void eval(const std::vector<array>& inputs, array& out);
  bool recip_;
};

class StopGradient : public UnaryPrimitive {
 public:
  explicit StopGradient(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(StopGradient)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Subtract : public UnaryPrimitive {
 public:
  explicit Subtract(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Subtract)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Tan : public UnaryPrimitive {
 public:
  explicit Tan(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Tan)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Tanh : public UnaryPrimitive {
 public:
  explicit Tanh(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Tanh)
  DEFINE_DEFAULT_IS_EQUIVALENT()
  DEFINE_INPUT_OUTPUT_SHAPE()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class Uniform : public UnaryPrimitive {
 public:
  explicit Uniform(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_PRINT(Uniform)
  DEFINE_DEFAULT_IS_EQUIVALENT()

 private:
  void eval(const std::vector<array>& inputs, array& out);
};

class View : public UnaryPrimitive {
 public:
  explicit View(Stream stream, Dtype dtype)
      : UnaryPrimitive(stream), dtype_(dtype) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  void print(std::ostream& os) override;
  bool is_equivalent(const Primitive& other) const override;

 private:
  Dtype dtype_;
};

class Transpose : public UnaryPrimitive {
 public:
  explicit Transpose(Stream stream, const std::vector<int>& axes)
      : UnaryPrimitive(stream), axes_(axes) {}

  void eval_cpu(const std::vector<array>& inputs, array& out) override;
  void eval_gpu(const std::vector<array>& inputs, array& out) override;

  DEFINE_GRADS()
  DEFINE_PRINT(Transpose)
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::vector<int> axes_;

  void eval(const std::vector<array>& inputs, array& out);
};

class QRF : public Primitive {
 public:
  explicit QRF(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(QRF)

 private:
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

}
