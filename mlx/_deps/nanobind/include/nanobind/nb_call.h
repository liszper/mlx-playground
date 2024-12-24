NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 6255)
#endif

class kwargs_proxy : public handle {
public:
    explicit kwargs_proxy(handle h) : handle(h) { }
};

class args_proxy : public handle {
public:
    explicit args_proxy(handle h) : handle(h) { }
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

template <typename Derived>
args_proxy api<Derived>::operator*() const {
    return args_proxy(derived().ptr());
}

template <typename T>
NB_INLINE void call_analyze(size_t &nargs, size_t &nkwargs, const T &value) {
    using D = std::decay_t<T>;
    static_assert(!std::is_base_of_v<arg_locked, D>, "nb::arg().lock() may be used only when defining functions, not when calling them");

    if constexpr (std::is_same_v<D, arg_v>)
        nkwargs++;
    else if constexpr (std::is_same_v<D, args_proxy>)
        nargs += len(value);
    else if constexpr (std::is_same_v<D, kwargs_proxy>)
        nkwargs += len(value);
    else
        nargs += 1;

    (void) nargs; (void) nkwargs; (void) value;
}

template <rv_policy policy, typename T>
NB_INLINE void call_init(PyObject **args, PyObject *kwnames, size_t &nargs,
                         size_t &nkwargs, const size_t kwargs_offset,
                         T &&value) {
    using D = std::decay_t<T>;

    if constexpr (std::is_same_v<D, arg_v>) {
        args[kwargs_offset + nkwargs] = value.value.release().ptr();
        NB_TUPLE_SET_ITEM(kwnames, nkwargs++,
                         PyUnicode_InternFromString(value.name_));
    } else if constexpr (std::is_same_v<D, args_proxy>) {
        for (size_t i = 0, l = len(value); i < l; ++i)
            args[nargs++] = borrow(value[i]).release().ptr();
    } else if constexpr (std::is_same_v<D, kwargs_proxy>) {
        PyObject *key, *entry;
        Py_ssize_t pos = 0;
        while (PyDict_Next(value.ptr(), &pos, &key, &entry)) {
            Py_INCREF(key); Py_INCREF(entry);
            args[kwargs_offset + nkwargs] = entry;
            NB_TUPLE_SET_ITEM(kwnames, nkwargs++, key);
        }
    } else {
        args[nargs++] = make_caster<T>::from_cpp((forward_t<T>) value, policy, nullptr).ptr();
    }
    (void) args; (void) kwnames; (void) nargs;
    (void) nkwargs; (void) kwargs_offset;
}

#define NB_DO_VECTORCALL()                                                     \
    PyObject *base, **args_p;                                                  \
    if constexpr (method_call) {                                               \
        base = derived().key().release().ptr();                                \
        args[0] = derived().base().inc_ref().ptr();                            \
        args_p = args;                                                         \
        nargs++;                                                               \
    } else {                                                                   \
        base = derived().inc_ref().ptr();                                      \
        args[0] = nullptr;                                                     \
        args_p = args + 1;                                                     \
    }                                                                          \
    nargs |= NB_VECTORCALL_ARGUMENTS_OFFSET;                                   \
    return steal(obj_vectorcall(base, args_p, nargs, kwnames, method_call))

template <typename Derived>
template <rv_policy policy, typename... Args>
object api<Derived>::operator()(Args &&...args_) const {
    static constexpr bool method_call =
        std::is_same_v<Derived, accessor<obj_attr>> ||
        std::is_same_v<Derived, accessor<str_attr>>;

    if constexpr (((std::is_same_v<Args, arg_v> ||
                    std::is_same_v<Args, args_proxy> ||
                    std::is_same_v<Args, kwargs_proxy>) || ...)) {
        size_t nargs = 0, nkwargs = 0, nargs2 = 0, nkwargs2 = 0;

        (call_analyze(nargs, nkwargs, (const Args &) args_), ...);

        PyObject **args = (PyObject **) alloca((nargs + nkwargs + 1) * sizeof(PyObject *));

        PyObject *kwnames = nkwargs ? PyTuple_New((Py_ssize_t) nkwargs) : nullptr;

        (call_init<policy>(args + 1, kwnames, nargs2, nkwargs2, nargs,
                           (forward_t<Args>) args_), ...);

        NB_DO_VECTORCALL();
    } else {
        PyObject *args[sizeof...(Args) + 1], *kwnames = nullptr;
        size_t nargs = 0;

        ((args[1 + nargs++] =
              detail::make_caster<Args>::from_cpp(
                  (detail::forward_t<Args>) args_, policy, nullptr)
                  .ptr()),
         ...);

        NB_DO_VECTORCALL();
    }
}

#undef NB_DO_VECTORCALL

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
