#pragma once

#include <nanobind/nanobind.h>
#include <utility>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T1, typename T2> struct type_caster<std::pair<T1, T2>> {
    using Value = std::pair<T1, T2>;

    using Caster1 = make_caster<T1>;
    using Caster2 = make_caster<T2>;

    template <typename T> using Cast = Value;

    static constexpr auto Name = const_name(NB_TYPING_TUPLE "[") + concat(Caster1::Name, Caster2::Name) + const_name("]");

    bool from_python(handle src, uint8_t flags,
                     cleanup_list *cleanup) noexcept {
        PyObject *temp;
        PyObject **o = seq_get_with_size(src.ptr(), 2, &temp);

        bool success = o &&
                       caster1.from_python(o[0], flags, cleanup) &&
                       caster2.from_python(o[1], flags, cleanup);

        Py_XDECREF(temp);

        return success;
    }

    template <typename T>
    static handle from_cpp(T *value, rv_policy policy, cleanup_list *cleanup) {
        if (!value)
            return none().release();
        return from_cpp(*value, policy, cleanup);
    }

    template <typename T>
    static handle from_cpp(T &&value, rv_policy policy,
                           cleanup_list *cleanup) noexcept {
        object o1 = steal(
            Caster1::from_cpp(forward_like_<T>(value.first), policy, cleanup));
        if (!o1.is_valid())
            return {};

        object o2 = steal(
            Caster2::from_cpp(forward_like_<T>(value.second), policy, cleanup));
        if (!o2.is_valid())
            return {};

        PyObject *r = PyTuple_New(2);
        NB_TUPLE_SET_ITEM(r, 0, o1.release().ptr());
        NB_TUPLE_SET_ITEM(r, 1, o2.release().ptr());
        return r;
    }

    template <typename T>
    bool can_cast() const noexcept {
        return caster1.template can_cast<T1>() && caster2.template can_cast<T2>();
    }

    explicit operator Value() {
        return Value(caster1.operator cast_t<T1>(),
                     caster2.operator cast_t<T2>());
    }

    Caster1 caster1;
    Caster2 caster2;
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
