#include <nanobind/nanobind.h>
#include <structmember.h>
#include "nb_internals.h"
#include <thread>

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#ifndef NB_INTERNALS_VERSION
#  define NB_INTERNALS_VERSION 15
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
#  define NB_BUILD_TYPE "_debug"
#else
#  define NB_BUILD_TYPE ""
#endif

#if defined(_MSC_VER)
#  define NB_COMPILER_TYPE "_msvc"
#elif defined(__INTEL_COMPILER)
#  define NB_COMPILER_TYPE "_icc"
#elif defined(__clang__)
#  define NB_COMPILER_TYPE "_clang"
#elif defined(__PGI)
#  define NB_COMPILER_TYPE "_pgi"
#elif defined(__MINGW32__)
#  define NB_COMPILER_TYPE "_mingw"
#elif defined(__CYGWIN__)
#  define NB_COMPILER_TYPE "_gcc_cygwin"
#elif defined(__GNUC__)
#  define NB_COMPILER_TYPE "_gcc"
#else
#  define NB_COMPILER_TYPE "_unknown"
#endif

#if defined(_LIBCPP_VERSION)
#  define NB_STDLIB "_libcpp"
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#  define NB_STDLIB "_libstdcpp"
#else
#  define NB_STDLIB ""
#endif

#if defined(__GXX_ABI_VERSION)
#  define NB_BUILD_ABI "_cxxabi" NB_TOSTRING(__GXX_ABI_VERSION)
#elif defined(_MSC_VER)
#  define NB_BUILD_ABI "_mscver" NB_TOSTRING(_MSC_VER)
#else
#  define NB_BUILD_ABI ""
#endif

#if defined(Py_LIMITED_API)
#  define NB_STABLE_ABI "_stable"
#else
#  define NB_STABLE_ABI ""
#endif

#if NB_VERSION_DEV > 0
  #define NB_VERSION_DEV_STR "_dev" NB_TOSTRING(NB_VERSION_DEV)
#else
  #define NB_VERSION_DEV_STR ""
#endif

#define NB_INTERNALS_ID                                                        \
    "v" NB_TOSTRING(NB_INTERNALS_VERSION)                                      \
        NB_VERSION_DEV_STR NB_COMPILER_TYPE NB_STDLIB NB_BUILD_ABI             \
            NB_BUILD_TYPE NB_STABLE_ABI

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

extern PyObject *nb_func_getattro(PyObject *, PyObject *);
extern PyObject *nb_func_get_doc(PyObject *, void *);
extern PyObject *nb_func_get_nb_signature(PyObject *, void *);
extern PyObject *nb_bound_method_getattro(PyObject *, PyObject *);
extern int nb_func_traverse(PyObject *, visitproc, void *);
extern int nb_func_clear(PyObject *);
extern void nb_func_dealloc(PyObject *);
extern int nb_bound_method_traverse(PyObject *, visitproc, void *);
extern int nb_bound_method_clear(PyObject *);
extern void nb_bound_method_dealloc(PyObject *);
extern PyObject *nb_method_descr_get(PyObject *, PyObject *, PyObject *);

#if PY_VERSION_HEX >= 0x03090000
#  define NB_HAVE_VECTORCALL_PY39_OR_NEWER NB_HAVE_VECTORCALL
#else
#  define NB_HAVE_VECTORCALL_PY39_OR_NEWER 0
#endif

static PyType_Slot nb_meta_slots[] = {
    { Py_tp_base, nullptr },
    { 0, nullptr }
};

static PyType_Spec nb_meta_spec = {
    /* .name = */ "nanobind.nb_meta",
    /* .basicsize = */ 0,
    /* .itemsize = */ 0,
    /* .flags = */ Py_TPFLAGS_DEFAULT,
    /* .slots = */ nb_meta_slots
};

static PyMemberDef nb_func_members[] = {
    { "__vectorcalloffset__", T_PYSSIZET,
      (Py_ssize_t) offsetof(nb_func, vectorcall), READONLY, nullptr },
    { nullptr, 0, 0, 0, nullptr }
};

static PyGetSetDef nb_func_getset[] = {
    { "__doc__", nb_func_get_doc, nullptr, nullptr, nullptr },
    { "__nb_signature__", nb_func_get_nb_signature, nullptr, nullptr, nullptr },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyType_Slot nb_func_slots[] = {
    { Py_tp_members, (void *) nb_func_members },
    { Py_tp_getset, (void *) nb_func_getset },
    { Py_tp_getattro, (void *) nb_func_getattro },
    { Py_tp_traverse, (void *) nb_func_traverse },
    { Py_tp_clear, (void *) nb_func_clear },
    { Py_tp_dealloc, (void *) nb_func_dealloc },
    { Py_tp_traverse, (void *) nb_func_traverse },
    { Py_tp_new, (void *) PyType_GenericNew },
    { Py_tp_call, (void *) PyVectorcall_Call },
    { 0, nullptr }
};

static PyType_Spec nb_func_spec = {
    /* .name = */ "nanobind.nb_func",
    /* .basicsize = */ (int) sizeof(nb_func),
    /* .itemsize = */ (int) sizeof(func_data),
    /* .flags = */ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                   NB_HAVE_VECTORCALL_PY39_OR_NEWER,
    /* .slots = */ nb_func_slots
};

static PyType_Slot nb_method_slots[] = {
    { Py_tp_members, (void *) nb_func_members },
    { Py_tp_getset, (void *) nb_func_getset },
    { Py_tp_getattro, (void *) nb_func_getattro },
    { Py_tp_traverse, (void *) nb_func_traverse },
    { Py_tp_clear, (void *) nb_func_clear },
    { Py_tp_dealloc, (void *) nb_func_dealloc },
    { Py_tp_descr_get, (void *) nb_method_descr_get },
    { Py_tp_new, (void *) PyType_GenericNew },
    { Py_tp_call, (void *) PyVectorcall_Call },
    { 0, nullptr }
};

static PyType_Spec nb_method_spec = {
    /*.name = */ "nanobind.nb_method",
    /*.basicsize = */ (int) sizeof(nb_func),
    /*.itemsize = */ (int) sizeof(func_data),
    /*.flags = */ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                  Py_TPFLAGS_METHOD_DESCRIPTOR |
                  NB_HAVE_VECTORCALL_PY39_OR_NEWER,
    /*.slots = */ nb_method_slots
};

static PyMemberDef nb_bound_method_members[] = {
    { "__vectorcalloffset__", T_PYSSIZET,
      (Py_ssize_t) offsetof(nb_bound_method, vectorcall), READONLY, nullptr },
    { "__func__", T_OBJECT_EX,
      (Py_ssize_t) offsetof(nb_bound_method, func), READONLY, nullptr },
    { "__self__", T_OBJECT_EX,
      (Py_ssize_t) offsetof(nb_bound_method, self), READONLY, nullptr },
    { nullptr, 0, 0, 0, nullptr }
};

static PyType_Slot nb_bound_method_slots[] = {
    { Py_tp_members, (void *) nb_bound_method_members },
    { Py_tp_getattro, (void *) nb_bound_method_getattro },
    { Py_tp_traverse, (void *) nb_bound_method_traverse },
    { Py_tp_clear, (void *) nb_bound_method_clear },
    { Py_tp_dealloc, (void *) nb_bound_method_dealloc },
    { Py_tp_traverse, (void *) nb_bound_method_traverse },
    { Py_tp_call, (void *) PyVectorcall_Call },
    { 0, nullptr }
};

static PyType_Spec nb_bound_method_spec = {
    /* .name = */ "nanobind.nb_bound_method",
    /* .basicsize = */ (int) sizeof(nb_bound_method),
    /* .itemsize = */ 0,
    /* .flags = */ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                   NB_HAVE_VECTORCALL_PY39_OR_NEWER,
    /* .slots = */ nb_bound_method_slots
};

void default_exception_translator(const std::exception_ptr &p, void *) {
    try {
        std::rethrow_exception(p);
    } catch (const std::bad_alloc &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
    } catch (const std::domain_error &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::length_error &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::out_of_range &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const std::range_error &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::overflow_error &e) {
        PyErr_SetString(PyExc_OverflowError, e.what());
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

nb_internals *internals = nullptr;
PyTypeObject *nb_meta_cache = nullptr;

static bool is_alive_value = false;
static bool *is_alive_ptr = &is_alive_value;
bool is_alive() noexcept { return *is_alive_ptr; }

static void internals_cleanup() {
    nb_internals *p = internals;
    if (!p)
        return;

    *is_alive_ptr = false;

#if !defined(PYPY_VERSION)
    bool print_leak_warnings = p->print_leak_warnings;

    size_t inst_leaks = 0, keep_alive_leaks = 0;

    for (size_t i = 0; i < p->shard_count; ++i) {
        nb_shard &s = p->shards[i];
        inst_leaks += s.inst_c2p.size();
        keep_alive_leaks += s.keep_alive.size();
    }

    bool leak = inst_leaks > 0 || keep_alive_leaks > 0;

    if (print_leak_warnings && inst_leaks > 0) {
        fprintf(stderr, "nanobind: leaked %zu instances!\n", inst_leaks);

#if !defined(Py_LIMITED_API)
        auto print_leak = [](void* k, PyObject* v) {
            type_data *tp = nb_type_data(Py_TYPE(v));
            fprintf(stderr, " - leaked instance %p of type \"%s\"\n", k, tp->name);
        };

        for (size_t i = 0; i < p->shard_count; ++i) {
            for (auto [k, v]: p->shards[i].inst_c2p) {
                if (NB_UNLIKELY(nb_is_seq(v))) {
                    nb_inst_seq* seq = nb_get_seq(v);
                    for(; seq != nullptr; seq = seq->next)
                        print_leak(k, seq->inst);
                } else {
                    print_leak(k, (PyObject*)v);
                }
            }
        }
#endif
    }

    if (print_leak_warnings && keep_alive_leaks > 0)
        fprintf(stderr, "nanobind: leaked %zu keep_alive records!\n",
                keep_alive_leaks);

#if !defined(NB_ABORT_ON_LEAK)
    if (!leak)
        print_leak_warnings = false;
#endif

    if (!p->type_c2p_slow.empty()) {
        if (print_leak_warnings) {
            fprintf(stderr, "nanobind: leaked %zu types!\n",
                    p->type_c2p_slow.size());
            int ctr = 0;
            for (const auto &kv : p->type_c2p_slow) {
                fprintf(stderr, " - leaked type \"%s\"\n", kv.second->name);
                if (ctr++ == 10) {
                    fprintf(stderr, " - ... skipped remainder\n");
                    break;
                }
            }
        }
        leak = true;
    }

    if (!p->funcs.empty()) {
        if (print_leak_warnings) {
            fprintf(stderr, "nanobind: leaked %zu functions!\n",
                    p->funcs.size());
            int ctr = 0;
            for (auto [f, p2] : p->funcs) {
                fprintf(stderr, " - leaked function \"%s\"\n",
                        nb_func_data(f)->name);
                if (ctr++ == 10) {
                    fprintf(stderr, " - ... skipped remainder\n");
                    break;
                }
            }
        }
        leak = true;
    }

    if (!leak) {
        nb_translator_seq* t = p->translators.next;
        while (t) {
            nb_translator_seq *next = t->next;
            delete t;
            t = next;
        }

        delete p;
        internals = nullptr;
        nb_meta_cache = nullptr;
    } else {
        if (print_leak_warnings) {
            fprintf(stderr, "nanobind: this is likely caused by a reference "
                            "counting issue in the binding code.\n");
        }

        #if defined(NB_ABORT_ON_LEAK)
            abort();
        #endif
    }
#endif
}

NB_NOINLINE void init(const char *name) {
    if (internals)
        return;

#if defined(PYPY_VERSION)
    PyObject *dict = PyEval_GetBuiltins();
#elif PY_VERSION_HEX < 0x03090000
    PyObject *dict = PyInterpreterState_GetDict(_PyInterpreterState_Get());
#else
    PyObject *dict = PyInterpreterState_GetDict(PyInterpreterState_Get());
#endif
    check(dict, "nanobind::detail::init(): could not access internals dictionary!");

    PyObject *key = PyUnicode_FromFormat("__nb_internals_%s_%s__",
                                         NB_INTERNALS_ID, name ? name : "");
    check(key, "nanobind::detail::init(): could not create dictionary key!");

    PyObject *capsule = dict_get_item_ref_or_fail(dict, key);
    if (capsule) {
        Py_DECREF(key);
        internals = (nb_internals *) PyCapsule_GetPointer(capsule, "nb_internals");
        check(internals,
              "nanobind::detail::internals_fetch(): capsule pointer is NULL!");
        nb_meta_cache = internals->nb_meta;
        is_alive_ptr = internals->is_alive_ptr;
        Py_DECREF(capsule);
        return;
    }

    nb_internals *p = new nb_internals();

    size_t shard_count = 1;
    p->shard_count = shard_count;

    str nb_name("nanobind");
    p->nb_module = PyModule_NewObject(nb_name.ptr());

    nb_meta_slots[0].pfunc = (PyObject *) &PyType_Type;
    nb_meta_cache = p->nb_meta = (PyTypeObject *) PyType_FromSpec(&nb_meta_spec);
    p->nb_type_dict = PyDict_New();
    p->nb_func = (PyTypeObject *) PyType_FromSpec(&nb_func_spec);
    p->nb_method = (PyTypeObject *) PyType_FromSpec(&nb_method_spec);
    p->nb_bound_method = (PyTypeObject *) PyType_FromSpec(&nb_bound_method_spec);

    for (size_t i = 0; i < shard_count; ++i) {
        p->shards[i].keep_alive.min_load_factor(.1f);
        p->shards[i].inst_c2p.min_load_factor(.1f);
    }

    check(p->nb_module && p->nb_meta && p->nb_type_dict && p->nb_func &&
              p->nb_method && p->nb_bound_method,
          "nanobind::detail::init(): initialization failed!");

#if PY_VERSION_HEX < 0x03090000
    p->nb_func->tp_flags |= NB_HAVE_VECTORCALL;
    p->nb_func->tp_vectorcall_offset = offsetof(nb_func, vectorcall);
    p->nb_method->tp_flags |= NB_HAVE_VECTORCALL;
    p->nb_method->tp_vectorcall_offset = offsetof(nb_func, vectorcall);
    p->nb_bound_method->tp_flags |= NB_HAVE_VECTORCALL;
    p->nb_bound_method->tp_vectorcall_offset = offsetof(nb_bound_method, vectorcall);
#endif

#if defined(Py_LIMITED_API)
    p->PyType_Type_tp_free = (freefunc) PyType_GetSlot(&PyType_Type, Py_tp_free);
    p->PyType_Type_tp_init = (initproc) PyType_GetSlot(&PyType_Type, Py_tp_init);
    p->PyType_Type_tp_dealloc = (destructor) PyType_GetSlot(&PyType_Type, Py_tp_dealloc);
    p->PyType_Type_tp_setattro = (setattrofunc) PyType_GetSlot(&PyType_Type, Py_tp_setattro);
    p->PyProperty_Type_tp_descr_get = (descrgetfunc) PyType_GetSlot(&PyProperty_Type, Py_tp_descr_get);
    p->PyProperty_Type_tp_descr_set = (descrsetfunc) PyType_GetSlot(&PyProperty_Type, Py_tp_descr_set);

    PyType_Slot dummy_slots[] = {
        { Py_tp_base, &PyType_Type },
        { 0, nullptr }
    };

    PyType_Spec dummy_spec = {
        /* .name = */ "nanobind.dummy",
        /* .basicsize = */ - (int) sizeof(void*),
        /* .itemsize = */ 0,
        /* .flags = */ Py_TPFLAGS_DEFAULT,
        /* .slots = */ dummy_slots
    };

    PyObject *dummy = PyType_FromMetaclass(
        p->nb_meta, p->nb_module, &dummy_spec, nullptr);
    p->type_data_offset = (uint8_t *) PyObject_GetTypeData(dummy, p->nb_meta) - (uint8_t *) dummy;
    Py_DECREF(dummy);
#endif

    p->translators = { default_exception_translator, nullptr, nullptr };
    is_alive_value = true;
    is_alive_ptr = &is_alive_value;
    p->is_alive_ptr = is_alive_ptr;

#if PY_VERSION_HEX < 0x030C0000 && !defined(PYPY_VERSION)
    /* The implementation of typing.py on CPython <3.12 tends to introduce
       spurious reference leaks that upset nanobind's leak checker. The
       following band-aid, installs an 'atexit' handler that clears LRU caches
       used in typing.py. To be resilient to potential future changes in
       typing.py, the implementation fails silently if any step goes wrong. For
       context, see https://github.com/python/cpython/issues/98253. */

    const char *str =
        "def cleanup():\n"
        "    try:\n"
        "        import sys\n"
        "        fs = getattr(sys.modules.get('typing'), '_cleanups', None)\n"
        "        if fs is not None:\n"
        "            for f in fs:\n"
        "                f()\n"
        "    except:\n"
        "        pass\n"
        "import atexit\n"
        "atexit.register(cleanup)\n"
        "del atexit, cleanup";

    PyObject *code = Py_CompileString(str, "<internal>", Py_file_input);
    if (code) {
        PyObject *result = PyEval_EvalCode(code, PyEval_GetGlobals(), nullptr);
        if (!result)
            PyErr_Clear();
        Py_XDECREF(result);
        Py_DECREF(code);
    } else {
        PyErr_Clear();
    }
#endif

    if (Py_AtExit(internals_cleanup))
        fprintf(stderr,
                "Warning: could not install the nanobind cleanup handler! This "
                "is needed to check for reference leaks and release remaining "
                "resources at interpreter shutdown (e.g., to avoid leaks being "
                "reported by tools like 'valgrind'). If you are a user of a "
                "python extension library, you can ignore this warning.");

    capsule = PyCapsule_New(p, "nb_internals", nullptr);
    int rv = PyDict_SetItem(dict, key, capsule);
    check(!rv && capsule,
          "nanobind::detail::init(): capsule creation failed!");
    Py_DECREF(capsule);
    Py_DECREF(key);
    internals = p;
}

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
