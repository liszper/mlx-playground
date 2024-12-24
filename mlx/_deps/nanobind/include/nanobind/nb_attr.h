NAMESPACE_BEGIN(NB_NAMESPACE)

struct scope {
    PyObject *value;
    NB_INLINE scope(handle value) : value(value.ptr()) {}
};

struct name {
    const char *value;
    NB_INLINE name(const char *value) : value(value) {}
};

struct arg_v;
struct arg_locked;
struct arg_locked_v;

struct arg {
    NB_INLINE constexpr explicit arg(const char *name = nullptr) : name_(name), signature_(nullptr) { }

    template <typename T> NB_INLINE arg_v operator=(T &&value) const;

    NB_INLINE arg &noconvert(bool value = true) {
        convert_ = !value;
        return *this;
    }
    NB_INLINE arg &none(bool value = true) {
        none_ = value;
        return *this;
    }
    NB_INLINE arg &sig(const char *value) {
        signature_ = value;
        return *this;
    }

    NB_INLINE arg_locked lock();

    const char *name_, *signature_;
    uint8_t convert_{ true };
    bool none_{ false };
};

struct arg_v : arg {
    object value;
    NB_INLINE arg_v(const arg &base, object &&value)
        : arg(base), value(std::move(value)) {}

  private:
    using arg::noconvert;
    using arg::none;
    using arg::sig;
    using arg::lock;
};

struct arg_locked : arg {
    NB_INLINE constexpr explicit arg_locked(const char *name = nullptr) : arg(name) { }
    NB_INLINE constexpr explicit arg_locked(const arg &base) : arg(base) { }

    template <typename T> NB_INLINE arg_locked_v operator=(T &&value) const;

    NB_INLINE arg_locked &noconvert(bool value = true) {
        convert_ = !value;
        return *this;
    }
    NB_INLINE arg_locked &none(bool value = true) {
        none_ = value;
        return *this;
    }
    NB_INLINE arg_locked &sig(const char *value) {
        signature_ = value;
        return *this;
    }

    NB_INLINE arg_locked &lock() { return *this; }
};

struct arg_locked_v : arg_locked {
    object value;
    NB_INLINE arg_locked_v(const arg_locked &base, object &&value)
        : arg_locked(base), value(std::move(value)) {}

  private:
    using arg_locked::noconvert;
    using arg_locked::none;
    using arg_locked::sig;
    using arg_locked::lock;
};

NB_INLINE arg_locked arg::lock() { return arg_locked{*this}; }

template <typename... Ts> struct call_guard {
    using type = detail::tuple<Ts...>;
};

struct dynamic_attr {};
struct is_weak_referenceable {};
struct is_method {};
struct is_implicit {};
struct is_operator {};
struct is_arithmetic {};
struct is_flag {};
struct is_final {};
struct is_generic {};
struct kw_only {};
struct lock_self {};

template <size_t /* Nurse */, size_t /* Patient */> struct keep_alive {};
template <typename T> struct supplement {};
template <typename T> struct intrusive_ptr {
    intrusive_ptr(void (*set_self_py)(T *, PyObject *) noexcept)
        : set_self_py(set_self_py) { }
    void (*set_self_py)(T *, PyObject *) noexcept;
};

struct type_slots {
    type_slots (const PyType_Slot *value) : value(value) { }
    const PyType_Slot *value;
};

struct type_slots_callback {
    using cb_t = void (*)(const detail::type_init_data *t,
                          PyType_Slot *&slots, size_t max_slots) noexcept;
    type_slots_callback(cb_t callback) : callback(callback) { }
    cb_t callback;
};

struct sig {
    const char *value;
    sig(const char *value) : value(value) { }
};

struct is_getter { };

NAMESPACE_BEGIN(literals)
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
NAMESPACE_END(literals)

NAMESPACE_BEGIN(detail)

enum class func_flags : uint32_t {
    /* Low 3 bits reserved for return value policy */

    has_name = (1 << 4),
    has_scope = (1 << 5),
    has_doc = (1 << 6),
    has_args = (1 << 7),
    has_var_args = (1 << 8),
    has_var_kwargs = (1 << 9),
    is_method = (1 << 10),
    is_constructor = (1 << 11),
    is_implicit = (1 << 12),
    is_operator = (1 << 13),
    has_free = (1 << 14),
    return_ref = (1 << 15),
    has_signature = (1 << 16),
    has_keep_alive = (1 << 17)
};

enum cast_flags : uint8_t {
    convert = (1 << 0),

    construct = (1 << 1),

    accepts_none = (1 << 2),

    manual = (1 << 3)
};

struct arg_data {
    const char *name;
    const char *signature;
    PyObject *name_py;
    PyObject *value;
    uint8_t flag;
};

template <size_t Size> struct func_data_prelim {
    void *capture[3];

    void (*free_capture)(void *);

    PyObject *(*impl)(void *, PyObject **, uint8_t *, rv_policy,
                      cleanup_list *);

    const char *descr;

    const std::type_info **descr_types;

    uint32_t flags;

    uint16_t nargs;

    uint16_t nargs_pos;

    const char *name;
    const char *doc;
    PyObject *scope;

#if defined(_MSC_VER)
    arg_data args[Size == 0 ? 1 : Size];
#else
    arg_data args[Size];
#endif
};

template <typename F>
NB_INLINE void func_extra_apply(F &f, const name &name, size_t &) {
    f.name = name.value;
    f.flags |= (uint32_t) func_flags::has_name;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const scope &scope, size_t &) {
    f.scope = scope.value;
    f.flags |= (uint32_t) func_flags::has_scope;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const sig &s, size_t &) {
    f.flags |= (uint32_t) func_flags::has_signature;
    f.name = s.value;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const char *doc, size_t &) {
    f.doc = doc;
    f.flags |= (uint32_t) func_flags::has_doc;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_method, size_t &) {
    f.flags |= (uint32_t) func_flags::is_method;
}

template <typename F>
NB_INLINE void func_extra_apply(F &, is_getter, size_t &) { }

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_implicit, size_t &) {
    f.flags |= (uint32_t) func_flags::is_implicit;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_operator, size_t &) {
    f.flags |= (uint32_t) func_flags::is_operator;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, rv_policy pol, size_t &) {
    f.flags = (f.flags & ~0b111) | (uint16_t) pol;
}

template <typename F>
NB_INLINE void func_extra_apply(F &, std::nullptr_t, size_t &) { }

template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg &a, size_t &index) {
    uint8_t flag = 0;
    if (a.none_)
        flag |= (uint8_t) cast_flags::accepts_none;
    if (a.convert_)
        flag |= (uint8_t) cast_flags::convert;

    arg_data &arg = f.args[index];
    arg.flag = flag;
    arg.name = a.name_;
    arg.signature = a.signature_;
    arg.value = nullptr;
    index++;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg_v &a, size_t &index) {
    arg_data &ad = f.args[index];
    func_extra_apply(f, (const arg &) a, index);
    ad.value = a.value.ptr();
}
template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg_locked_v &a, size_t &index) {
    arg_data &ad = f.args[index];
    func_extra_apply(f, (const arg_locked &) a, index);
    ad.value = a.value.ptr();
}

template <typename F>
NB_INLINE void func_extra_apply(F &, kw_only, size_t &) {}

template <typename F>
NB_INLINE void func_extra_apply(F &, lock_self, size_t &) {}

template <typename F, typename... Ts>
NB_INLINE void func_extra_apply(F &, call_guard<Ts...>, size_t &) {}

template <typename F, size_t Nurse, size_t Patient>
NB_INLINE void func_extra_apply(F &f, nanobind::keep_alive<Nurse, Patient>, size_t &) {
    f.flags |= (uint32_t) func_flags::has_keep_alive;
}

template <typename... Ts> struct func_extra_info {
    using call_guard = void;
    static constexpr bool keep_alive = false;
    static constexpr size_t nargs_locked = 0;
};

template <typename T, typename... Ts> struct func_extra_info<T, Ts...>
    : func_extra_info<Ts...> { };

template <typename... Cs, typename... Ts>
struct func_extra_info<nanobind::call_guard<Cs...>, Ts...> : func_extra_info<Ts...> {
    static_assert(std::is_same_v<typename func_extra_info<Ts...>::call_guard, void>, "call_guard<> can only be specified once!");
    using call_guard = nanobind::call_guard<Cs...>;
};

template <size_t Nurse, size_t Patient, typename... Ts>
struct func_extra_info<nanobind::keep_alive<Nurse, Patient>, Ts...> : func_extra_info<Ts...> {
    static constexpr bool keep_alive = true;
};

template <typename... Ts>
struct func_extra_info<nanobind::arg_locked, Ts...> : func_extra_info<Ts...> {
    static constexpr size_t nargs_locked = 1 + func_extra_info<Ts...>::nargs_locked;
};

template <typename... Ts>
struct func_extra_info<nanobind::lock_self, Ts...> : func_extra_info<Ts...> {
    static constexpr size_t nargs_locked = 1 + func_extra_info<Ts...>::nargs_locked;
};

template <typename T>
NB_INLINE void process_keep_alive(PyObject **, PyObject *, T *) { }

template <size_t Nurse, size_t Patient>
NB_INLINE void
process_keep_alive(PyObject **args, PyObject *result,
                   nanobind::keep_alive<Nurse, Patient> *) {
    keep_alive(Nurse   == 0 ? result : args[Nurse - 1],
               Patient == 0 ? result : args[Patient - 1]);
}

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
