#pragma once

#include <nanobind/nanobind.h>
#include <memory>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

struct py_deleter {
    void operator()(void *) noexcept {
        if (!Py_IsInitialized())
            return;
        gil_scoped_acquire guard;
        Py_DECREF(o);
    }

    PyObject *o;
};

template <typename T>
inline NB_NOINLINE std::shared_ptr<T>
shared_from_python(T *ptr, handle h) noexcept {
    if (ptr)
        return std::shared_ptr<T>(ptr, py_deleter{ h.inc_ref().ptr() });
    else
        return std::shared_ptr<T>(nullptr);
}

inline NB_NOINLINE void shared_from_cpp(std::shared_ptr<void> &&ptr,
                                        PyObject *o) noexcept {
    keep_alive(o, new std::shared_ptr<void>(std::move(ptr)),
               [](void *p) noexcept { delete (std::shared_ptr<void> *) p; });
}

template <typename T> struct type_caster<std::shared_ptr<T>> {
    static constexpr bool IsClass = true;
    using Caster = make_caster<T>;
    using Td = std::decay_t<T>;

    NB_TYPE_CASTER(std::shared_ptr<T>, Caster::Name)

    static_assert(is_base_caster_v<Caster>,
                  "Conversion of ``shared_ptr<T>`` requires that ``T`` is "
                  "handled by nanobind's regular class binding mechanism. "
                  "However, a type caster was registered to intercept this "
                  "particular type, which is not allowed.");

    bool from_python(handle src, uint8_t flags,
                     cleanup_list *cleanup) noexcept {
        Caster caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;

        Td *ptr = caster.operator Td *();
        if constexpr (has_shared_from_this_v<T>) {
            if (ptr) {
                if (auto sp = ptr->weak_from_this().lock()) {
                    value = std::static_pointer_cast<T>(std::move(sp));
                    return true;
                }
            }
            value = shared_from_python(ptr, src);
        } else {
            value = std::static_pointer_cast<T>(
                shared_from_python(static_cast<void *>(ptr), src));
        }
        return true;
    }

    static handle from_cpp(const Value &value, rv_policy,
                           cleanup_list *cleanup) noexcept {
        bool is_new = false;
        handle result;

        Td *ptr = (Td *) value.get();
        const std::type_info *type = &typeid(Td);

        constexpr bool has_type_hook = !std::is_base_of_v<std::false_type, type_hook<Td>>;
        if constexpr (has_type_hook)
            type = type_hook<Td>::get(ptr);

        if constexpr (!std::is_polymorphic_v<Td>) {
            result = nb_type_put(type, ptr, rv_policy::reference,
                                 cleanup, &is_new);
        } else {
            const std::type_info *type_p = (!has_type_hook && ptr) ? &typeid(*ptr) : nullptr;

            result = nb_type_put_p(type, type_p, ptr, rv_policy::reference,
                                   cleanup, &is_new);
        }

        if (is_new) {
            std::shared_ptr<void> pp;
            if constexpr (std::is_const_v<T>)
                pp = std::static_pointer_cast<void>(std::const_pointer_cast<Td>(value));
            else
                pp = std::static_pointer_cast<void>(value);
            shared_from_cpp(std::move(pp), result.ptr());
        }

        return result;
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
