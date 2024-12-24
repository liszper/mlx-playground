NAMESPACE_BEGIN(NB_NAMESPACE)

#if PY_VERSION_HEX >= 0x030C0000
struct error_scope {
    error_scope() { value = PyErr_GetRaisedException(); }
    ~error_scope() { PyErr_SetRaisedException(value); }
private:
    PyObject *value;
};
#else
struct error_scope {
    error_scope() { PyErr_Fetch(&type, &value, &trace); }
    ~error_scope() { PyErr_Restore(type, value, trace); }
private:
    PyObject *type, *value, *trace;
};
#endif

class NB_EXPORT python_error : public std::exception {
public:
    NB_EXPORT_SHARED python_error();
    NB_EXPORT_SHARED python_error(const python_error &);
    NB_EXPORT_SHARED python_error(python_error &&) noexcept;
    NB_EXPORT_SHARED ~python_error() override;

    bool matches(handle exc) const noexcept {
#if PY_VERSION_HEX < 0x030C0000
        return PyErr_GivenExceptionMatches(m_type, exc.ptr()) != 0;
#else
        return PyErr_GivenExceptionMatches(m_value, exc.ptr()) != 0;
#endif
    }

    NB_EXPORT_SHARED void restore() noexcept;

    void discard_as_unraisable(handle context) noexcept {
        restore();
        PyErr_WriteUnraisable(context.ptr());
    }

    void discard_as_unraisable(const char *context) noexcept {
        object context_s = steal(PyUnicode_FromString(context));
        discard_as_unraisable(context_s);
    }

    handle value() const { return m_value; }

#if PY_VERSION_HEX < 0x030C0000
    handle type() const { return m_type; }
    object traceback() const { return borrow(m_traceback); }
#else
    handle type() const { return value().type(); }
    object traceback() const { return steal(PyException_GetTraceback(m_value)); }
#endif
    [[deprecated]]
    object trace() const { return traceback(); }

    NB_EXPORT_SHARED const char *what() const noexcept override;

private:
#if PY_VERSION_HEX < 0x030C0000
    mutable PyObject *m_type = nullptr;
    mutable PyObject *m_value = nullptr;
    mutable PyObject *m_traceback = nullptr;
#else
    mutable PyObject *m_value = nullptr;
#endif
    mutable char *m_what = nullptr;
};

using cast_error = std::bad_cast;

enum class exception_type {
    runtime_error, stop_iteration, index_error, key_error, value_error,
    type_error, buffer_error, import_error, attribute_error, next_overload
};

class NB_EXPORT builtin_exception : public std::runtime_error {
public:
    NB_EXPORT_SHARED builtin_exception(exception_type type, const char *what);
    NB_EXPORT_SHARED builtin_exception(builtin_exception &&) = default;
    NB_EXPORT_SHARED builtin_exception(const builtin_exception &) = default;
    NB_EXPORT_SHARED ~builtin_exception();
    NB_EXPORT_SHARED exception_type type() const { return m_type; }
private:
    exception_type m_type;
};

#define NB_EXCEPTION(name)                                                     \
    inline builtin_exception name(const char *what = nullptr) {                \
        return builtin_exception(exception_type::name, what);                  \
    }

NB_EXCEPTION(stop_iteration)
NB_EXCEPTION(index_error)
NB_EXCEPTION(key_error)
NB_EXCEPTION(value_error)
NB_EXCEPTION(type_error)
NB_EXCEPTION(buffer_error)
NB_EXCEPTION(import_error)
NB_EXCEPTION(attribute_error)
NB_EXCEPTION(next_overload)

#undef NB_EXCEPTION

inline void register_exception_translator(detail::exception_translator t,
                                          void *payload = nullptr) {
    detail::register_exception_translator(t, payload);
}

template <typename T>
class exception : public object {
    NB_OBJECT_DEFAULT(exception, object, "Exception", PyExceptionClass_Check)

    exception(handle scope, const char *name, handle base = PyExc_Exception)
        : object(detail::exception_new(scope.ptr(), name, base.ptr()),
                 detail::steal_t()) {
        detail::register_exception_translator(
            [](const std::exception_ptr &p, void *payload) {
                try {
                    std::rethrow_exception(p);
                } catch (T &e) {
                    PyErr_SetString((PyObject *) payload, e.what());
                }
            }, m_ptr);
    }
};

NB_CORE void chain_error(handle type, const char *fmt, ...) noexcept;
[[noreturn]] NB_CORE void raise_from(python_error &e, handle type, const char *fmt, ...);

NAMESPACE_END(NB_NAMESPACE)
