#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr uint8_t MAGIC[] = {
    0x93,
    0x4e,
    0x55,
    0x4d,
    0x50,
    0x59,
};

inline bool is_big_endian() {
  union ByteOrder {
    int32_t i;
    uint8_t c[4];
  };
  ByteOrder b = {0x01234567};

  return b.c[0] == 0x01;
}

std::string dtype_to_array_protocol(const Dtype& t) {
  std::ostringstream r;
  if (size_of(t) > 1) {
    r << (is_big_endian() ? ">" : "<");
  } else {
    r << "|";
  }
  r << kindof(t) << (int)size_of(t);
  return r.str();
}

Dtype dtype_from_array_protocol(std::string_view t) {
  if (t.length() == 2 || t.length() == 3) {
    std::string_view r = t.length() == 3 ? t.substr(1, 2) : t;

    if (r == "V2") {
      return bfloat16;
    }

    uint8_t size = r[1] - '0';

    switch (r[0]) {
      case 'b': {
        if (size == 1)
          return bool_;
      }
      case 'i': {
        if (size == 1)
          return int8;
        else if (size == 2)
          return int16;
        else if (size == 4)
          return int32;
        else if (size == 8)
          return int64;
      }
      case 'u': {
        if (size == 1)
          return uint8;
        else if (size == 2)
          return uint16;
        else if (size == 4)
          return uint32;
        else if (size == 8)
          return uint64;
      }
      case 'f': {
        if (size == 2)
          return float16;
        else if (size == 4)
          return float32;
      }
    }
  }

  throw std::invalid_argument("[from_str] Invalid array protocol type-string: " + std::string(t));
}

}

void save(std::shared_ptr<io::Writer> out_stream, array a) {

  a.eval();

  if (a.nbytes() == 0) {
    throw std::invalid_argument("[save] cannot serialize an empty array");
  }

  if (!(a.flags().row_contiguous || a.flags().col_contiguous)) {
    a = reshape(flatten(a), a.shape());
    a.eval();
  }
  if (!(a.flags().row_contiguous || a.flags().col_contiguous)) {
    throw std::invalid_argument("[save] can only serialize row or col contiguous arrays");
  }

  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error("[save] Failed to open " + out_stream->label());
  }

  std::ostringstream magic_ver_len;
  magic_ver_len.write(reinterpret_cast<const char*>(MAGIC), 6);

  std::string fortran_order = a.flags().col_contiguous ? "True" : "False";
  std::ostringstream header;
  header << "{'descr': '" << dtype_to_array_protocol(a.dtype()) << "',"
         << " 'fortran_order': " << fortran_order << "," << " 'shape': (";
  for (auto i : a.shape()) {
    header << i << ", ";
  }
  header << ")}";

  size_t header_len = static_cast<size_t>(header.tellp());
  bool is_v1 = header_len + 15 < std::numeric_limits<uint16_t>::max();

  size_t padding = (6 + 2 + (2 + 2 * is_v1) + header_len + 1) % 16;

  header << std::string(padding, ' ') << '\n';

  if (is_v1) {
    magic_ver_len << (char)0x01 << (char)0x00;

    uint16_t v1_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v1_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 2);
    } else {
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  } else {
    magic_ver_len << (char)0x02 << (char)0x00;

    uint32_t v2_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v2_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 4);
    } else {
      magic_ver_len.write(len_bytes + 3, 1);
      magic_ver_len.write(len_bytes + 2, 1);
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  }

  out_stream->write(magic_ver_len.str().c_str(), magic_ver_len.str().length());
  out_stream->write(header.str().c_str(), header.str().length());
  out_stream->write(a.data<char>(), a.nbytes());
}

void save(std::string file, array a) {
  if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npy")
    file += ".npy";

  save(std::make_shared<io::FileWriter>(std::move(file)), a);
}

array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s) {
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error("[load] Failed to open " + in_stream->label());
  }

  char read_magic_and_ver[8];
  in_stream->read(read_magic_and_ver, 8);
  if (std::memcmp(read_magic_and_ver, MAGIC, 6) != 0) {
    throw std::runtime_error("[load] Invalid header in " + in_stream->label());
  }

  if (read_magic_and_ver[6] != 1 && read_magic_and_ver[6] != 2) {
    throw std::runtime_error("[load] Unsupported npy format version in " + in_stream->label());
  }

  int header_len_size = read_magic_and_ver[6] == 1 ? 2 : 4;
  size_t header_len;

  if (header_len_size == 2) {
    uint16_t v1_header_len;
    in_stream->read(reinterpret_cast<char*>(&v1_header_len), header_len_size);
    header_len = v1_header_len;
  } else {
    uint32_t v2_header_len;
    in_stream->read(reinterpret_cast<char*>(&v2_header_len), header_len_size);
    header_len = v2_header_len;
  }

  std::vector<char> buffer(header_len + 1);
  in_stream->read(&buffer[0], header_len);
  buffer[header_len] = 0;
  std::string header(&buffer[0]);

  std::string dtype_str = header.substr(11, 3);
  bool read_is_big_endian = dtype_str[0] == '>';
  Dtype dtype = dtype_from_array_protocol(dtype_str);

  bool col_contiguous = header[34] == 'T';

  std::vector<int> shape;

  size_t st = header.find_last_of('(') + 1;
  size_t ed = header.find_last_of(')');
  std::string shape_str = header.substr(st, ed - st);

  while (!shape_str.empty()) {
    size_t pos;
    int dim = std::stoi(shape_str, &pos);
    shape.push_back(dim);

    if (pos + 2 <= shape_str.length())
      shape_str = shape_str.substr(pos + 2);
    else {
      shape_str = shape_str.substr(pos);
      if (!shape_str.empty() && shape_str != " " && shape_str != ",") {
        throw std::runtime_error("[load] Unknown error while parsing header in " +
            in_stream->label());
      }
      shape_str = "";
    }
  }

  size_t offset = 8 + header_len_size + header.length();
  bool swap_endianness = read_is_big_endian != is_big_endian();

  if (col_contiguous) {
    std::reverse(shape.begin(), shape.end());
  }
  auto loaded_array = array(
      shape,
      dtype,
      std::make_shared<Load>(to_stream(s), in_stream, offset, swap_endianness),
      std::vector<array>{});
  if (col_contiguous) {
    loaded_array = transpose(loaded_array, s);
  }

  return loaded_array;
}

array load(std::string file, StreamOrDevice s) {
  return load(std::make_shared<io::ParallelFileReader>(std::move(file)), s);
}

namespace io {

ThreadPool& thread_pool() {
  static ThreadPool pool_{4};
  return pool_;
}

ThreadPool ParallelFileReader::thread_pool_{4};

void ParallelFileReader::read(char* data, size_t n) {
  while (n != 0) {
    auto m = ::read(fd_, data, std::min(n, static_cast<size_t>(INT32_MAX)));
    if (m <= 0) {
      std::ostringstream msg;
      msg << "[read] Unable to read " << n << " bytes from file.";
      throw std::runtime_error(msg.str());
    }
    data += m;
    n -= m;
  }
}

void ParallelFileReader::read(char* data, size_t n, size_t offset) {
  auto readfn = [fd = fd_](size_t offset, size_t size, char* buffer) -> bool {
    while (size != 0) {
      auto m = pread(fd, buffer, size, offset);
      if (m <= 0) {
        return false;
      }
      buffer += m;
      size -= m;
    }
    return true;
  };
  std::vector<std::future<bool>> futs;
  while (n != 0) {
    if (n < batch_size_) {
      if (!readfn(offset, n, data)) {
        throw std::runtime_error("[read] Unable to read from file.");
      }
      break;
    } else {
      size_t m = batch_size_;
      futs.emplace_back(thread_pool_.enqueue(readfn, offset, m, data));
      data += m;
      n -= m;
      offset += m;
    }
  }
  for (auto& f : futs) {
    if (!f.get()) {
      throw std::runtime_error("[read] Unable to read from file.");
    }
  }
}

}

}
