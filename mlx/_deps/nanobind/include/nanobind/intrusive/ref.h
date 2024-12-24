#pragma once

#include "counter.h"

NAMESPACE_BEGIN(nanobind)

template <typename T> class ref {
public:
    ref() = default;

    ref(T *ptr) : m_ptr(ptr) { inc_ref((intrusive_base *) m_ptr); }

    ref(const ref &r) : m_ptr(r.m_ptr) { inc_ref((intrusive_base *) m_ptr); }

    ref(ref &&r) noexcept : m_ptr(r.m_ptr) { r.m_ptr = nullptr; }

    ~ref() { dec_ref((intrusive_base *) m_ptr); }

    ref &operator=(ref &&r) noexcept {
        dec_ref((intrusive_base *) m_ptr);
        m_ptr = r.m_ptr;
        r.m_ptr = nullptr;
        return *this;
    }

    ref &operator=(const ref &r) {
        inc_ref((intrusive_base *) r.m_ptr);
        dec_ref((intrusive_base *) m_ptr);
        m_ptr = r.m_ptr;
        return *this;
    }

    ref &operator=(T *ptr) {
        inc_ref((intrusive_base *) ptr);
        dec_ref((intrusive_base *) m_ptr);
        m_ptr = ptr;
        return *this;
    }

    void reset() {
        dec_ref((intrusive_base *) m_ptr);
        m_ptr = nullptr;
    }

    bool operator==(const ref &r) const { return m_ptr == r.m_ptr; }

    bool operator!=(const ref &r) const { return m_ptr != r.m_ptr; }

    bool operator==(const T *ptr) const { return m_ptr == ptr; }

    bool operator!=(const T *ptr) const { return m_ptr != ptr; }

    T *operator->() { return m_ptr; }

    const T *operator->() const { return m_ptr; }

    T &operator*() { return *m_ptr; }

    const T &operator*() const { return *m_ptr; }

    operator T *() { return m_ptr; }

    operator const T *() const { return m_ptr; }

    T *get() { return m_ptr; }

    const T *get() const { return m_ptr; }

private:
    T *m_ptr = nullptr;
};

#if defined(NB_VERSION_MAJOR)
NAMESPACE_BEGIN(detail)
template <typename T> struct type_caster<nanobind::ref<T>> {
    using Caster = make_caster<T>;
    static constexpr bool IsClass = true;
    NB_TYPE_CASTER(ref<T>, Caster::Name)

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        Caster caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;

        value = Value(caster.operator T *());
        return true;
    }

    static handle from_cpp(const ref<T> &value, rv_policy policy, cleanup_list *cleanup) noexcept {
        if constexpr (std::is_base_of_v<intrusive_base, T>)
            if (policy != rv_policy::copy && policy != rv_policy::move && value.get())
                if (PyObject* obj = value->self_py())
                    return handle(obj).inc_ref();

        return Caster::from_cpp(value.get(), policy, cleanup);
    }
};
NAMESPACE_END(detail)
#endif

NAMESPACE_END(nanobind)
