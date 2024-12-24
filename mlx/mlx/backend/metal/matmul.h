#include "mlx/backend/metal/device.h"

namespace mlx::core {

void steel_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    std::vector<int> batch_shape = {},
    std::vector<size_t> A_batch_stride = {},
    std::vector<size_t> B_batch_stride = {});

}
