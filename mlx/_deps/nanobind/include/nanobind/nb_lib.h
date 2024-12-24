NAMESPACE_BEGIN(NB_NAMESPACE)

namespace dlpack { struct dltensor; struct dtype; }

NAMESPACE_BEGIN(detail)

struct ndarray_handle;
struct ndarray_config;

struct NB_CORE cleanup_list {
public:
    static constexpr uint32_t Small = 6;

    cleanup_list(PyObject *self) :
        m_size{1},
        m_capacity{Small},
        m_data{m_local} {
        m_local[0] = self;
    }

    ~cleanup_list() = default;

    NB_INLINE void append(PyObject *value) noexcept {
        if (m_size >= m_capacity)
            expand();
        m_data[m_size++] = value;
    }

    NB_INLINE PyObject *self() const {
        return m_local[0];
    }

    void release() noexcept;

    bool used() { return m_size != 1; }

    size_t size() const { return m_size; }

    PyObject *operator[](size_t index) const { return m_data[index]; }

protected:
    void expand() noexcept;

protected:
    uint32_t m_size;
    uint32_t m_capacity;
    PyObject **m_data;
    PyObject *m_local[Small];
};

#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
NB_CORE void raise(const char *fmt, ...);

#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
NB_CORE void raise_type_error(const char *fmt, ...);

#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
NB_CORE void fail(const char *fmt, ...) noexcept;

[[noreturn]] NB_CORE void raise_python_error();

NB_CORE void raise_next_overload_if_null(void *p);

[[noreturn]] NB_CORE void raise_cast_error();

NB_CORE void init(const char *domain);

NB_CORE PyObject *str_from_obj(PyObject *o);

NB_CORE PyObject *str_from_cstr(const char *c);

NB_CORE PyObject *str_from_cstr_and_size(const char *c, size_t n);

NB_CORE PyObject *bytes_from_obj(PyObject *o);

NB_CORE PyObject *bytes_from_cstr(const char *c);

NB_CORE PyObject *bytes_from_cstr_and_size(const void *c, size_t n);

NB_CORE PyObject *bytearray_from_obj(PyObject *o);

NB_CORE PyObject *bytearray_from_cstr_and_size(const void *c, size_t n);

NB_CORE PyObject *bool_from_obj(PyObject *o);

NB_CORE PyObject *int_from_obj(PyObject *o);

NB_CORE PyObject *float_from_obj(PyObject *o);

NB_CORE PyObject *list_from_obj(PyObject *o);

NB_CORE PyObject *tuple_from_obj(PyObject *o);

NB_CORE PyObject *set_from_obj(PyObject *o);

NB_CORE PyObject *getattr(PyObject *obj, const char *key);
NB_CORE PyObject *getattr(PyObject *obj, PyObject *key);

NB_CORE PyObject *getattr(PyObject *obj, const char *key, PyObject *def) noexcept;
NB_CORE PyObject *getattr(PyObject *obj, PyObject *key, PyObject *def) noexcept;

NB_CORE void getattr_or_raise(PyObject *obj, const char *key, PyObject **out);
NB_CORE void getattr_or_raise(PyObject *obj, PyObject *key, PyObject **out);

NB_CORE void setattr(PyObject *obj, const char *key, PyObject *value);
NB_CORE void setattr(PyObject *obj, PyObject *key, PyObject *value);

NB_CORE void delattr(PyObject *obj, const char *key);
NB_CORE void delattr(PyObject *obj, PyObject *key);

NB_CORE void getitem_or_raise(PyObject *obj, Py_ssize_t, PyObject **out);
NB_CORE void getitem_or_raise(PyObject *obj, const char *key, PyObject **out);
NB_CORE void getitem_or_raise(PyObject *obj, PyObject *key, PyObject **out);

NB_CORE void setitem(PyObject *obj, Py_ssize_t, PyObject *value);
NB_CORE void setitem(PyObject *obj, const char *key, PyObject *value);
NB_CORE void setitem(PyObject *obj, PyObject *key, PyObject *value);

NB_CORE void delitem(PyObject *obj, Py_ssize_t);
NB_CORE void delitem(PyObject *obj, const char *key);
NB_CORE void delitem(PyObject *obj, PyObject *key);

NB_CORE size_t obj_len(PyObject *o);

NB_CORE size_t obj_len_hint(PyObject *o) noexcept;

NB_CORE PyObject* obj_repr(PyObject *o);

NB_CORE bool obj_comp(PyObject *a, PyObject *b, int value);

NB_CORE PyObject *obj_op_1(PyObject *a, PyObject* (*op)(PyObject*));

NB_CORE PyObject *obj_op_2(PyObject *a, PyObject *b,
                           PyObject *(*op)(PyObject *, PyObject *));

NB_CORE PyObject *obj_vectorcall(PyObject *base, PyObject *const *args,
                                 size_t nargsf, PyObject *kwnames,
                                 bool method_call);

NB_CORE PyObject *obj_iter(PyObject *o);

NB_CORE PyObject *obj_iter_next(PyObject *o);

NB_CORE void tuple_check(PyObject *tuple, size_t nargs);

NB_CORE void call_append_arg(PyObject *args, size_t &nargs, PyObject *value);

NB_CORE void call_append_args(PyObject *args, size_t &nargs, PyObject *value);

NB_CORE void call_append_kwarg(PyObject *kwargs, const char *name, PyObject *value);

NB_CORE void call_append_kwargs(PyObject *kwargs, PyObject *value);

NB_CORE PyObject **seq_get_with_size(PyObject *seq, size_t size,
                                     PyObject **temp) noexcept;

NB_CORE PyObject **seq_get(PyObject *seq, size_t *size,
                           PyObject **temp) noexcept;

NB_CORE PyObject *capsule_new(const void *ptr, const char *name,
                              void (*cleanup)(void *) noexcept) noexcept;

NB_CORE PyObject *nb_func_new(const void *data) noexcept;

struct type_init_data;
NB_CORE PyObject *nb_type_new(const type_init_data *c) noexcept;

NB_CORE bool nb_type_get(const std::type_info *t, PyObject *o, uint8_t flags,
                         cleanup_list *cleanup, void **out) noexcept;

NB_CORE PyObject *nb_type_put(const std::type_info *cpp_type, void *value,
                              rv_policy rvp, cleanup_list *cleanup,
                              bool *is_new = nullptr) noexcept;

NB_CORE PyObject *nb_type_put_p(const std::type_info *cpp_type,
                                const std::type_info *cpp_type_p, void *value,
                                rv_policy rvp, cleanup_list *cleanup,
                                bool *is_new = nullptr) noexcept;

NB_CORE PyObject *nb_type_put_unique(const std::type_info *cpp_type,
                                     void *value, cleanup_list *cleanup,
                                     bool cpp_delete) noexcept;

NB_CORE PyObject *nb_type_put_unique_p(const std::type_info *cpp_type,
                                       const std::type_info *cpp_type_p,
                                       void *value, cleanup_list *cleanup,
                                       bool cpp_delete) noexcept;

NB_CORE bool nb_type_relinquish_ownership(PyObject *o, bool cpp_delete) noexcept;

NB_CORE void nb_type_restore_ownership(PyObject *o, bool cpp_delete) noexcept;

NB_CORE void *nb_type_supplement(PyObject *t) noexcept;

NB_CORE bool nb_type_check(PyObject *t) noexcept;

NB_CORE size_t nb_type_size(PyObject *t) noexcept;

NB_CORE size_t nb_type_align(PyObject *t) noexcept;

NB_CORE PyObject *nb_type_name(PyObject *t) noexcept;

NB_CORE PyObject *nb_inst_name(PyObject *o) noexcept;

NB_CORE const std::type_info *nb_type_info(PyObject *t) noexcept;

NB_CORE void *nb_inst_ptr(PyObject *o) noexcept;

NB_CORE bool nb_type_isinstance(PyObject *obj, const std::type_info *t) noexcept;

NB_CORE PyObject *nb_type_lookup(const std::type_info *t) noexcept;

NB_CORE PyObject *nb_inst_alloc(PyTypeObject *t);

NB_CORE PyObject *nb_inst_alloc_zero(PyTypeObject *t);

NB_CORE PyObject *nb_inst_reference(PyTypeObject *t, void *ptr,
                                    PyObject *parent);

NB_CORE PyObject *nb_inst_take_ownership(PyTypeObject *t, void *ptr);

NB_CORE void nb_inst_destruct(PyObject *o) noexcept;

NB_CORE void nb_inst_zero(PyObject *o) noexcept;

NB_CORE void nb_inst_copy(PyObject *dst, const PyObject *src) noexcept;

NB_CORE void nb_inst_move(PyObject *dst, const PyObject *src) noexcept;

NB_CORE void nb_inst_replace_copy(PyObject *dst, const PyObject *src) noexcept;

NB_CORE void nb_inst_replace_move(PyObject *dst, const PyObject *src) noexcept;

NB_CORE bool nb_inst_python_derived(PyObject *o) noexcept;

NB_CORE void nb_inst_set_state(PyObject *o, bool ready, bool destruct) noexcept;

NB_CORE std::pair<bool, bool> nb_inst_state(PyObject *o) noexcept;

NB_CORE void property_install(PyObject *scope, const char *name,
                              PyObject *getter, PyObject *setter) noexcept;

NB_CORE void property_install_static(PyObject *scope, const char *name,
                                     PyObject *getter,
                                     PyObject *setter) noexcept;

NB_CORE PyObject *get_override(void *ptr, const std::type_info *type,
                               const char *name, bool pure);

NB_CORE void keep_alive(PyObject *nurse, PyObject *patient);

NB_CORE void keep_alive(PyObject *nurse, void *payload,
                        void (*deleter)(void *) noexcept) noexcept;

NB_CORE void implicitly_convertible(const std::type_info *src,
                                    const std::type_info *dst) noexcept;

NB_CORE void implicitly_convertible(bool (*predicate)(PyTypeObject *,
                                                      PyObject *,
                                                      cleanup_list *),
                                    const std::type_info *dst) noexcept;

struct enum_init_data;

NB_CORE PyObject *enum_create(enum_init_data *) noexcept;

NB_CORE void enum_append(PyObject *tp, const char *name,
                         int64_t value, const char *doc) noexcept;

NB_CORE bool enum_from_python(const std::type_info *, PyObject *, int64_t *,
                              uint8_t flags) noexcept;

NB_CORE PyObject *enum_from_cpp(const std::type_info *, int64_t) noexcept;

NB_CORE void enum_export(PyObject *tp);

NB_CORE PyObject *module_import(const char *name);

NB_CORE PyObject *module_import(PyObject *name);

NB_CORE PyObject *module_new(const char *name, PyModuleDef *def) noexcept;

NB_CORE PyObject *module_new_submodule(PyObject *base, const char *name,
                                       const char *doc) noexcept;

NB_CORE ndarray_handle *ndarray_import(PyObject *o,
                                       const ndarray_config *c,
                                       bool convert,
                                       cleanup_list *cleanup) noexcept;

NB_CORE ndarray_handle *ndarray_create(void *value, size_t ndim,
                                       const size_t *shape, PyObject *owner,
                                       const int64_t *strides,
                                       dlpack::dtype dtype, bool ro,
                                       int device, int device_id,
                                       char order);

NB_CORE dlpack::dltensor *ndarray_inc_ref(ndarray_handle *) noexcept;

NB_CORE void ndarray_dec_ref(ndarray_handle *) noexcept;

NB_CORE PyObject *ndarray_export(ndarray_handle *, int framework,
                                 rv_policy policy, cleanup_list *cleanup) noexcept;

NB_CORE bool ndarray_check(PyObject *o) noexcept;

NB_CORE void print(PyObject *file, PyObject *str, PyObject *end);

typedef void (*exception_translator)(const std::exception_ptr &, void *);

NB_CORE void register_exception_translator(exception_translator translator,
                                           void *payload);

NB_CORE PyObject *exception_new(PyObject *mod, const char *name,
                                PyObject *base);

NB_CORE bool load_i8 (PyObject *o, uint8_t flags, int8_t *out) noexcept;
NB_CORE bool load_u8 (PyObject *o, uint8_t flags, uint8_t *out) noexcept;
NB_CORE bool load_i16(PyObject *o, uint8_t flags, int16_t *out) noexcept;
NB_CORE bool load_u16(PyObject *o, uint8_t flags, uint16_t *out) noexcept;
NB_CORE bool load_i32(PyObject *o, uint8_t flags, int32_t *out) noexcept;
NB_CORE bool load_u32(PyObject *o, uint8_t flags, uint32_t *out) noexcept;
NB_CORE bool load_i64(PyObject *o, uint8_t flags, int64_t *out) noexcept;
NB_CORE bool load_u64(PyObject *o, uint8_t flags, uint64_t *out) noexcept;
NB_CORE bool load_f32(PyObject *o, uint8_t flags, float *out) noexcept;
NB_CORE bool load_f64(PyObject *o, uint8_t flags, double *out) noexcept;

NB_CORE void incref_checked(PyObject *o) noexcept;

NB_CORE void decref_checked(PyObject *o) noexcept;

NB_CORE bool leak_warnings() noexcept;
NB_CORE bool implicit_cast_warnings() noexcept;
NB_CORE void set_leak_warnings(bool value) noexcept;
NB_CORE void set_implicit_cast_warnings(bool value) noexcept;

NB_CORE bool iterable_check(PyObject *o) noexcept;

NB_CORE void slice_compute(PyObject *slice, Py_ssize_t size,
                           Py_ssize_t &start, Py_ssize_t &stop,
                           Py_ssize_t &step, size_t &slice_length);

NB_CORE bool issubclass(PyObject *a, PyObject *b);

NB_CORE PyObject *repr_list(PyObject *o);
NB_CORE PyObject *repr_map(PyObject *o);

NB_CORE bool is_alive() noexcept;

#if NB_TYPE_GET_SLOT_IMPL
NB_CORE void *type_get_slot(PyTypeObject *t, int slot_id);
#endif

NB_CORE PyObject *dict_get_item_ref_or_fail(PyObject *d, PyObject *k);

NAMESPACE_END(detail)

using detail::raise;
using detail::raise_type_error;
using detail::raise_python_error;

NAMESPACE_END(NB_NAMESPACE)
