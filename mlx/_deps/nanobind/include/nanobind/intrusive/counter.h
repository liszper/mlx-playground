#pragma once

#include <cstdint>

#if !defined(NB_INTRUSIVE_EXPORT)
#  define NB_INTRUSIVE_EXPORT
#endif

#if !defined(Py_PYTHON_H)
extern "C" {
    struct _object;
    typedef _object PyObject;
};
#endif

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

NAMESPACE_BEGIN(nanobind)

struct NB_INTRUSIVE_EXPORT intrusive_counter {
public:
    intrusive_counter() noexcept = default;

    intrusive_counter(const intrusive_counter &) noexcept { }
    intrusive_counter(intrusive_counter &&) noexcept { }
    intrusive_counter &operator=(const intrusive_counter &) noexcept { return *this; }
    intrusive_counter &operator=(intrusive_counter &&) noexcept { return *this; }

    void inc_ref() const noexcept;

    bool dec_ref() const noexcept;

    PyObject *self_py() const noexcept;

    void set_self_py(PyObject *self) noexcept;

protected:
    mutable uintptr_t m_state = 1;
};

static_assert(sizeof(intrusive_counter) == sizeof(void *), "The intrusive_counter class should always have the same size as a pointer.");

class NB_INTRUSIVE_EXPORT intrusive_base {
public:
    void inc_ref() const noexcept { m_ref_count.inc_ref(); }

    bool dec_ref() const noexcept { return m_ref_count.dec_ref(); }

    void set_self_py(PyObject *self) noexcept { m_ref_count.set_self_py(self); }

    PyObject *self_py() const noexcept { return m_ref_count.self_py(); }

    virtual ~intrusive_base() = default;

private:
    mutable intrusive_counter m_ref_count;
};

inline void inc_ref(const intrusive_base *o) noexcept {
    if (o)
        o->inc_ref();
}

inline void dec_ref(const intrusive_base *o) noexcept {
    if (o && o->dec_ref())
        delete o;
}

extern NB_INTRUSIVE_EXPORT
void intrusive_init(void (*intrusive_inc_ref_py)(PyObject *) noexcept,
                    void (*intrusive_dec_ref_py)(PyObject *) noexcept);

NAMESPACE_END(nanobind)
