#include <nanobind/stl/vector.h>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

bool is_istream_object(const nb::object& file) {
  return nb::hasattr(file, "readinto") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_ostream_object(const nb::object& file) {
  return nb::hasattr(file, "write") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_zip_file(const nb::module_& zipfile, const nb::object& file) {
  if (is_istream_object(file)) {
    auto st_pos = file.attr("tell")();
    bool r = nb::cast<bool>(zipfile.attr("is_zipfile")(file));
    file.attr("seek")(st_pos, 0);
    return r;
  }
  return nb::cast<bool>(zipfile.attr("is_zipfile")(file));
}

class ZipFileWrapper {
 public:
  ZipFileWrapper(
      const nb::module_& zipfile,
      const nb::object& file,
      char mode = 'r',
      int compression = 0)
      : zipfile_module_(zipfile),
        zipfile_object_(zipfile.attr("ZipFile")(
            file,
            "mode"_a = mode,
            "compression"_a = compression,
            "allowZip64"_a = true)),
        files_list_(zipfile_object_.attr("namelist")()),
        open_func_(zipfile_object_.attr("open")),
        read_func_(zipfile_object_.attr("read")),
        close_func_(zipfile_object_.attr("close")) {}

  std::vector<std::string> namelist() const {
    return nb::cast<std::vector<std::string>>(files_list_);
  }

  nb::object open(const std::string& key, char mode = 'r') {
    if (mode == 'w') {
      return open_func_(key, "mode"_a = mode, "force_zip64"_a = true);
    }
    return open_func_(key, "mode"_a = mode);
  }

 private:
  nb::module_ zipfile_module_;
  nb::object zipfile_object_;
  nb::list files_list_;
  nb::object open_func_;
  nb::object read_func_;
  nb::object close_func_;
};

class PyFileReader : public io::Reader {
 public:
  PyFileReader(nb::object file)
      : pyistream_(file),
        readinto_func_(file.attr("readinto")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileReader() {
    nb::gil_scoped_acquire gil;

    pyistream_.release().dec_ref();
    readinto_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyistream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyistream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void read(char* data, size_t n) override {
    nb::gil_scoped_acquire gil;
    _read(data, n);
  }

  void read(char* data, size_t n, size_t offset) override {
    nb::gil_scoped_acquire gil;
    seek_func_(offset, (int)std::ios_base::beg);
    _read(data, n);
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  void _read(char* data, size_t n) {
    auto memview = PyMemoryView_FromMemory(data, n, PyBUF_WRITE);
    nb::object bytes_read = readinto_func_(nb::handle(memview));

    if (bytes_read.is_none() || nb::cast<size_t>(bytes_read) < n) {
      throw std::runtime_error("[load] Failed to read from python stream");
    }
  }

  nb::object pyistream_;
  nb::object readinto_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};

class PyFileWriter : public io::Writer {
 public:
  PyFileWriter(nb::object file)
      : pyostream_(file),
        write_func_(file.attr("write")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileWriter() {
    nb::gil_scoped_acquire gil;

    pyostream_.release().dec_ref();
    write_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyostream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyostream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void write(const char* data, size_t n) override {
    nb::gil_scoped_acquire gil;

    auto memview = PyMemoryView_FromMemory(const_cast<char*>(data), n, PyBUF_READ);
    nb::object bytes_written = write_func_(nb::handle(memview));

    if (bytes_written.is_none() || nb::cast<size_t>(bytes_written) < n) {
      throw std::runtime_error("[load] Failed to write to python stream");
    }
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  nb::object pyostream_;
  nb::object write_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};
