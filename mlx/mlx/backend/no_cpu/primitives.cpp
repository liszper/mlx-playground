#include "mlx/primitives.h"

#define NO_CPU_MULTI(func)                                             \
  void func::eval_cpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CPU implementation.");     \
  }

#define NO_CPU(func)                                                  \
  void func::eval_cpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CPU implementation.");    \
  }

namespace mlx::core {

NO_CPU(Abs)
NO_CPU(Add)
NO_CPU(Arange)
NO_CPU(ArgReduce)
NO_CPU(AsType)
NO_CPU(AsStrided)
NO_CPU(BitwiseBinary)
NO_CPU(Broadcast)
NO_CPU(Ceil)
NO_CPU(Concatenate)
NO_CPU(Copy)
NO_CPU(Cos)
NO_CPU(Cosh)
NO_CPU_MULTI(Depends)
NO_CPU(Divide)
NO_CPU_MULTI(DivMod)
NO_CPU(NumberOfElements)
NO_CPU(Remainder)
NO_CPU(Equal)
NO_CPU(Erf)
NO_CPU(ErfInv)
NO_CPU(Exp)
NO_CPU(Expm1)
NO_CPU(Floor)
NO_CPU(Full)
NO_CPU(Gather)
NO_CPU(Greater)
NO_CPU(GreaterEqual)
NO_CPU(Less)
NO_CPU(LessEqual)
NO_CPU(Load)
NO_CPU(Log)
NO_CPU(Log1p)
NO_CPU(LogicalNot)
NO_CPU(LogicalAnd)
NO_CPU(LogicalOr)
NO_CPU(Matmul)
NO_CPU(Maximum)
NO_CPU(Minimum)
NO_CPU(Multiply)
NO_CPU(Negative)
NO_CPU(NotEqual)
NO_CPU(Pad)
NO_CPU(Power)
NO_CPU_MULTI(QRF)
NO_CPU(RandomBits)
NO_CPU(Reduce)
NO_CPU(Reshape)
NO_CPU(Round)
NO_CPU(Scan)
NO_CPU(Scatter)
NO_CPU(Select)
NO_CPU(Sigmoid)
NO_CPU(Sign)
NO_CPU(Sin)
NO_CPU(Sinh)
NO_CPU(Slice)
NO_CPU(SliceUpdate)
NO_CPU(Softmax)
NO_CPU_MULTI(Split)
NO_CPU(Square)
NO_CPU(Sqrt)
NO_CPU(StopGradient)
NO_CPU(Subtract)
NO_CPU(Tan)
NO_CPU(Tanh)
NO_CPU(Transpose)
NO_CPU(View)

}
