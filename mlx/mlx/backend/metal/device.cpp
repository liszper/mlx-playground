#include <cstdlib>
#include <sstream>

#include <sys/sysctl.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::metal {

namespace {

constexpr int MAX_BUFFERS_PER_QUEUE = 12;
constexpr int MAX_DISPATCHES_PER_ENCODER = 2;

constexpr const char* default_mtllib_path = METAL_PATH;

constexpr auto get_metal_version() {
#if (MLX_METAL_VERSION >= 320)
  return MTL::LanguageVersion3_2;
#elif (MLX_METAL_VERSION >= 310)
  return MTL::LanguageVersion3_1;
#else
  return MTL::LanguageVersion3_0;
#endif
}

auto load_device() {
  auto devices = MTL::CopyAllDevices();
  auto device = static_cast<MTL::Device*>(devices->object(0))
      ?: MTL::CreateSystemDefaultDevice();
  if (!device) {
    throw std::runtime_error("Failed to load device");
  }
  return device;
}
std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  NS::Error* error;
  auto lib = device->newLibrary(library, &error);

  return std::make_pair(lib, error);
}

#ifdef SWIFTPM_BUNDLE
MTL::Library* try_load_bundle(MTL::Device* device, NS::URL* url) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        "default.metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}
#endif

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name = "mlx",
    const char* lib_path = default_mtllib_path) {
  std::string first_path = get_colocated_mtllib_path(lib_name);
  if (first_path.size() != 0) {
    auto [lib, error] = load_library_from_path(device, first_path.c_str());
    if (lib) {
      return lib;
    }
  }

#ifdef SWIFTPM_BUNDLE
  {
    MTL::Library* library = try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL());
    if (library != nullptr) {
      return library;
    }
    auto bundles = NS::Bundle::allBundles();
    for (int i = 0, c = (int)bundles->count(); i < c; i++) {
      auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
      library = try_load_bundle(device, bundle->resourceURL());
      if (library != nullptr) {
        return library;
      }
    }
  }
#endif

  {
    auto [lib, error] = load_library_from_path(device, lib_path);
    if (!lib) {
      std::ostringstream msg;
      msg << error->localizedDescription()->utf8String() << "\n"
          << "Failed to load device library from <" << lib_path << ">"
          << " or <" << first_path << ">.";
      throw std::runtime_error(msg.str());
    }
    return lib;
  }
}

}

CommandEncoder::CommandEncoder(MTL::CommandBuffer* cbuf) : cbuf(cbuf) {
  enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  enc->retain();
}

CommandEncoder::~CommandEncoder() {
  enc->endEncoding();
  enc->release();
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (auto it = outputs.find(r_buf); it != outputs.end()) {
    enc->memoryBarrier(&r_buf, 1);

    outputs.erase(it);
  }
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto base_offset = a.data<char>() -
      static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
  base_offset += offset;
  enc->setBuffer(a_buf, base_offset, idx);
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  set_input_array(a, idx, offset);
  auto buf = static_cast<MTL::Resource*>(a.buffer().ptr());
  if (concurrent) {
    concurrent_outputs.insert(buf);
  } else {
    outputs.insert(buf);
  }
}

void CommandEncoder::dispatchThreadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  num_dispatches++;
  enc->dispatchThreadgroups(grid_dims, group_dims);
  maybe_split();
}

void CommandEncoder::dispatchThreads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  num_dispatches++;
  enc->dispatchThreads(grid_dims, group_dims);
  maybe_split();
}

void CommandEncoder::maybe_split() {
  if (num_dispatches > MAX_DISPATCHES_PER_ENCODER && !concurrent) {
    enc->endEncoding();
    enc->release();
    num_dispatches = 0;
    outputs.clear();
    enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    enc->retain();
  }
}

Device::Device() {
  auto pool = new_scoped_memory_pool();
  device_ = load_device();
  library_map_ = {{"mlx", load_library(device_)}};
}

Device::~Device() {
  auto pool = new_scoped_memory_pool();
  for (auto& q : queue_map_) {
    q.second->release();
  }
  for (auto& b : buffer_map_) {
    b.second.second->release();
  }
  for (auto& k : kernel_map_) {
    k.second->release();
  }
  for (auto& l : library_map_) {
    l.second->release();
  }
  device_->release();
}

void Device::new_queue(int index) {
  auto thread_pool = metal::new_scoped_memory_pool();

  const std::lock_guard<std::mutex> lock(mtx_);
  auto q = device_->newCommandQueue(MAX_BUFFERS_PER_QUEUE);
  debug_set_stream_queue_label(q, index);
  if (!q) {
    throw std::runtime_error("[metal::Device] Failed to make new command queue.");
  }
  queue_map_.insert({index, q});
}

int Device::get_command_buffer_ops(int index) {
  auto bit = buffer_map_.find(index);
  return bit->second.first;
}

void Device::increment_command_buffer_ops(int index) {
  auto bit = buffer_map_.find(index);
  bit->second.first++;
}

MTL::CommandBuffer* Device::get_command_buffer(int index) {
  auto bit = buffer_map_.find(index);
  if (bit == buffer_map_.end()) {
    auto qit = queue_map_.find(index);
    if (qit == queue_map_.end()) {
      throw std::runtime_error("[metal::Device] Attempting to get command buffer for invalid queue.");
    }

    auto cb = qit->second->commandBufferWithUnretainedReferences();

    if (!cb) {
      throw std::runtime_error("[metal::Device] Unable to create new command buffer");
    }

    cb->retain();

    bit = buffer_map_.insert({index, {0, cb}}).first;
  }
  return bit->second.second;
}

void Device::commit_command_buffer(int index) {
  auto bit = buffer_map_.find(index);
  bit->second.second->commit();
  bit->second.second->release();
  buffer_map_.erase(bit);
}

void Device::end_encoding(int index) {
  encoder_map_.erase(index);
}

CommandEncoder& Device::get_command_encoder(int index) {
  auto eit = encoder_map_.find(index);
  if (eit == encoder_map_.end()) {
    auto cb = get_command_buffer(index);
    eit = encoder_map_.emplace(index, std::make_unique<CommandEncoder>(cb)).first;
  }
  return *(eit->second);
}

void Device::register_library(
    const std::string& lib_name,
    const std::string& lib_path) {
  if (auto it = library_map_.find(lib_name); it == library_map_.end()) {
    auto new_lib = load_library(device_, lib_name, lib_path.c_str());
    library_map_.insert({lib_name, new_lib});
  }
}

MTL::Library* Device::get_library_cache_(const std::string& lib_name) {
  MTL::Library* mtl_lib;
  if (auto it = library_map_.find(lib_name); it != library_map_.end()) {
    mtl_lib = it->second;
  } else {
    register_library(lib_name, get_colocated_mtllib_path(lib_name));
    mtl_lib = library_map_[lib_name];
  }

  return mtl_lib;
}

MTL::Library* Device::get_library_(const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code = NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
  auto mtl_lib = device_->newLibrary(ns_code, options, &error);
  options->release();

  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source" << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

MTL::Function* Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = mtl_lib->newFunction(ns_name);

  return mtl_function;
}

MTL::Function* Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  auto mtl_func_consts = MTL::FunctionConstantValues::alloc()->init();

  for (auto [value, type, index] : func_consts) {
    mtl_func_consts->setConstantValue(value, type, index);
  }

  auto desc = MTL::FunctionDescriptor::functionDescriptor();
  desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
  desc->setSpecializedName(
      NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
  desc->setConstantValues(mtl_func_consts);

  NS::Error* error = nullptr;
  auto mtl_function = mtl_lib->newFunction(desc, &error);

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  mtl_func_consts->release();

  return mtl_function;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  NS::Error* error = nullptr;
  MTL::ComputePipelineState* kernel;

  if (mtl_function) {
    kernel = device_->newComputePipelineState(mtl_function, &error);
  }

  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function,
    const MTL::LinkedFunctions* linked_functions) {
  if (!linked_functions) {
    return get_kernel_(name, mtl_function);
  }

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    throw std::runtime_error(msg.str());
  }

  auto desc = MTL::ComputePipelineDescriptor::alloc()->init();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);

  NS::Error* error = nullptr;
  auto kernel = device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error);

  if (!kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::Library* Device::get_library(const std::string& name) {
  auto it = library_map_.find(name);
  return (it != library_map_.end()) ? it->second : nullptr;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& source,
    bool cache /* = true */) {
  if (cache) {
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second;
    }
  }

  auto mtl_lib = get_library_(source);

  if (cache) {
    library_map_.insert({name, mtl_lib});
  }

  return mtl_lib;
}

MTL::Function* Device::get_function(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& specialized_name /* = "" */,
    const MTLFCList& func_consts /* = {} */) {
  return get_function_(base_name, specialized_name, func_consts, mtl_lib);
}

MTL::Function* Device::get_function(
    const std::string& base_name,
    const std::string& lib_name /* = "mlx" */,
    const std::string& specialized_name /*  = "" */,
    const MTLFCList& func_consts /* = {} */) {
  MTL::Library* mtl_lib = get_library_cache_(lib_name);

  return get_function(base_name, mtl_lib, specialized_name, func_consts);
}

MTL::LinkedFunctions* Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto lfuncs = MTL::LinkedFunctions::linkedFunctions();

  std::vector<NS::Object*> objs(funcs.size());
  for (int i = 0; i < funcs.size(); i++) {
    objs[i] = funcs[i];
  }

  NS::Array* funcs_arr = NS::Array::array(objs.data(), funcs.size());

  lfuncs->setPrivateFunctions(funcs_arr);

  return lfuncs;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  auto pool = new_scoped_memory_pool();

  const auto& kname = hash_name.empty() ? base_name : hash_name;
  if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
    return it->second;
  }

  auto mtl_function = get_function_(base_name, kname, func_consts, mtl_lib);

  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel = get_kernel_(kname, mtl_function, mtl_linked_funcs);

  mtl_function->release();
  mtl_linked_funcs->release();

  kernel_map_.insert({kname, kernel});

  return kernel;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& lib_name /* = "mlx" */,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  const auto& kname = hash_name.size() == 0 ? base_name : hash_name;
  if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
    return it->second;
  }

  MTL::Library* mtl_lib = get_library_cache_(lib_name);

  return get_kernel(base_name, mtl_lib, kname, func_consts, linked_functions);
}

Device& device(mlx::core::Device) {
  static Device metal_device;
  return metal_device;
}

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool() {
  auto dtor = [](void* ptr) {
    static_cast<NS::AutoreleasePool*>(ptr)->release();
  };
  return std::unique_ptr<void, std::function<void(void*)>>(
      NS::AutoreleasePool::alloc()->init(), dtor);
}

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    device(stream.device).new_queue(stream.index);
  }
}

std::unordered_map<std::string, std::variant<std::string, size_t>>
device_info() {
  auto raw_device = device(default_device()).mtl_device();
  auto arch = std::string(raw_device->architecture()->name()->utf8String());

  int mib[] = {CTL_HW, HW_MEMSIZE};
  size_t memsize = 0;
  size_t length = sizeof(memsize);

  sysctl(mib, 2, &memsize, &length, NULL, 0);

  return {
      {"architecture", arch},
      {"max_buffer_length", raw_device->maxBufferLength()},
      {"max_recommended_working_set_size", raw_device->recommendedMaxWorkingSetSize()},
      {"memory_size", memsize}};
}

}
