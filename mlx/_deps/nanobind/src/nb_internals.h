#pragma once

#if defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#elif defined(_MSC_VER)
#  pragma warning(disable: 4127)
#  pragma warning(disable: 4324)
#  pragma warning(disable: 4293)
#  pragma warning(disable: 4310)
#endif

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>
#include <cstring>
#include <string_view>
#include <functional>
#include "hash.h"

#if TSL_RH_VERSION_MAJOR != 1 || TSL_RH_VERSION_MINOR < 3
#  error nanobind depends on tsl::robin_map, in particular version >= 1.3.0, <2.0.0
#endif

#if defined(_MSC_VER)
#  define NB_THREAD_LOCAL __declspec(thread)
#else
#  define NB_THREAD_LOCAL __thread
#endif

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

#define check(cond, ...) if (NB_UNLIKELY(!(cond))) nanobind::detail::fail(__VA_ARGS__)

struct func_data : func_data_prelim<0> {
    arg_data *args;
    char *signature;
};

struct nb_inst {
    PyObject_HEAD

    int32_t offset;

    uint32_t state : 2;

    static constexpr uint32_t state_uninitialized = 0;
    static constexpr uint32_t state_relinquished = 1;
    static constexpr uint32_t state_ready = 2;

    uint32_t direct : 1;

    uint32_t internal : 1;

    uint32_t destruct : 1;

    uint32_t cpp_delete : 1;

    uint32_t clear_keep_alive : 1;

    uint32_t intrusive : 1;

    uint32_t unused : 24;
};

static_assert(sizeof(nb_inst) == sizeof(PyObject) + sizeof(uint32_t) * 2);

struct nb_func {
    PyObject_VAR_HEAD
    PyObject* (*vectorcall)(PyObject *, PyObject * const*, size_t, PyObject *);
    uint32_t max_nargs;
    bool complex_call;
    bool doc_uniform;
};

struct nb_ndarray {
    PyObject_HEAD
    ndarray_handle *th;
};

struct nb_bound_method {
    PyObject_HEAD
    PyObject* (*vectorcall)(PyObject *, PyObject * const*, size_t, PyObject *);
    nb_func *func;
    PyObject *self;
};

struct ptr_hash {
    size_t operator()(const void *p) const {
        if constexpr (sizeof(void *) == 4)
            return (size_t) fmix32((uint32_t) (uintptr_t) p);
        else
            return (size_t) fmix64((uint64_t) (uintptr_t) p);
    }
};

template <typename T> class py_allocator {
public:
    using value_type = T;
    using pointer = T *;
    using size_type = std::size_t;

    py_allocator() = default;
    py_allocator(const py_allocator &) = default;

    template <typename U> py_allocator(const py_allocator<U> &) { }

    pointer allocate(size_type n, const void * /*hint*/ = nullptr) noexcept {
        void *p = PyMem_Malloc(n * sizeof(T));
        if (!p)
            fail("PyMem_Malloc(): out of memory!");
        return static_cast<pointer>(p);
    }

    void deallocate(T *p, size_type /*n*/) noexcept { PyMem_Free(p); }
};

struct nb_inst_seq {
    PyObject *inst;
    nb_inst_seq *next;
};

struct nb_alias_chain {
    const std::type_info *value;
    nb_alias_chain *next;
};

struct nb_weakref_seq {
    void (*callback)(void *) noexcept;
    void *payload;
    nb_weakref_seq *next;
};

struct std_typeinfo_hash {
    size_t operator()(const std::type_info *a) const {
        const char *name = a->name();
        return std::hash<std::string_view>()({name, strlen(name)});
    }
};

struct std_typeinfo_eq {
    bool operator()(const std::type_info *a, const std::type_info *b) const {
        return a->name() == b->name() || strcmp(a->name(), b->name()) == 0;
    }
};

using nb_type_map_fast = tsl::robin_map<const std::type_info *, type_data *, ptr_hash>;
using nb_type_map_slow = tsl::robin_map<const std::type_info *, type_data *,
                                        std_typeinfo_hash, std_typeinfo_eq>;

using nb_ptr_map  = tsl::robin_map<void *, void*, ptr_hash>;

NB_INLINE bool         nb_is_seq(void *p)   { return ((uintptr_t) p) & 1; }

NB_INLINE void*        nb_mark_seq(void *p) { return (void *) (((uintptr_t) p) | 1); }

NB_INLINE nb_inst_seq* nb_get_seq(void *p)  { return (nb_inst_seq *) (((uintptr_t) p) ^ 1); }

struct nb_translator_seq {
    exception_translator translator;
    void *payload;
    nb_translator_seq *next = nullptr;
};

struct nb_shard {
    nb_ptr_map inst_c2p;

    nb_ptr_map keep_alive;
};

struct nb_internals {
    PyObject *nb_module;

    PyTypeObject *nb_meta;

    PyObject *nb_type_dict;

    PyTypeObject *nb_func, *nb_method, *nb_bound_method;

    PyTypeObject *nb_static_property = nullptr;
    descrsetfunc nb_static_property_descr_set = nullptr;

    bool nb_static_property_disabled = false;

    PyTypeObject *nb_ndarray = nullptr;

    nb_shard shards[1];
    inline nb_shard &shard(void *) { return shards[0]; }

    nb_type_map_fast type_c2p_fast;

    nb_type_map_slow type_c2p_slow;

    nb_ptr_map funcs;

    nb_translator_seq translators;

    bool print_leak_warnings = true;

    bool print_implicit_cast_warnings = true;

    bool *is_alive_ptr = nullptr;

#if defined(Py_LIMITED_API)
    freefunc PyType_Type_tp_free;
    initproc PyType_Type_tp_init;
    destructor PyType_Type_tp_dealloc;
    setattrofunc PyType_Type_tp_setattro;
    descrgetfunc PyProperty_Type_tp_descr_get;
    descrsetfunc PyProperty_Type_tp_descr_set;
    size_t type_data_offset;
#endif

    size_t shard_count = 1;
};

#if defined(Py_LIMITED_API)
#  define NB_SLOT(type, name) internals->type##_##name
#else
#  define NB_SLOT(type, name) type.name
#endif

extern nb_internals *internals;
extern PyTypeObject *nb_meta_cache;

extern char *type_name(const std::type_info *t);

extern PyObject *inst_new_ext(PyTypeObject *tp, void *value);
extern PyObject *inst_new_int(PyTypeObject *tp, PyObject *args, PyObject *kwds);
extern PyTypeObject *nb_static_property_tp() noexcept;
extern type_data *nb_type_c2p(nb_internals *internals,
                              const std::type_info *type);
extern void nb_type_unregister(type_data *t) noexcept;

extern PyObject *call_one_arg(PyObject *fn, PyObject *arg) noexcept;

NB_INLINE func_data *nb_func_data(void *o) {
    return (func_data *) (((char *) o) + sizeof(nb_func));
}

#if defined(Py_LIMITED_API)
extern type_data *nb_type_data_static(PyTypeObject *o) noexcept;
#endif

NB_INLINE type_data *nb_type_data(PyTypeObject *o) noexcept{
    #if !defined(Py_LIMITED_API)
        return (type_data *) (((char *) o) + sizeof(PyHeapTypeObject));
    #else
        return nb_type_data_static(o);
    #endif
}

extern PyObject *nb_type_name(PyObject *o) noexcept;

inline void *inst_ptr(nb_inst *self) {
    void *ptr = (void *) ((intptr_t) self + self->offset);
    return self->direct ? ptr : *(void **) ptr;
}

template <typename T> struct scoped_pymalloc {
    scoped_pymalloc(size_t size = 1) {
        ptr = (T *) PyMem_Malloc(size * sizeof(T));
        if (!ptr)
            fail("scoped_pymalloc(): could not allocate %zu bytes of memory!", size);
    }
    ~scoped_pymalloc() { PyMem_Free(ptr); }
    T *release() {
        T *temp = ptr;
        ptr = nullptr;
        return temp;
    }
    T *get() const { return ptr; }
    T &operator[](size_t i) { return ptr[i]; }
    T *operator->() { return ptr; }
private:
    T *ptr{ nullptr };
};

struct lock_shard { lock_shard(nb_shard &) { } };
struct lock_internals { lock_internals(nb_internals *) { } };
struct unlock_internals { unlock_internals(nb_internals *) { } };
struct lock_obj { lock_obj(PyObject *) { } };

extern char *strdup_check(const char *);
extern void *malloc_check(size_t size);
extern void maybe_make_immortal(PyObject *op);

extern char *extract_name(const char *cmd, const char *prefix, const char *s);

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
