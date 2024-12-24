#pragma once

#include <nanobind/nanobind.h>

NAMESPACE_BEGIN(NB_NAMESPACE)

enum eval_mode {
    eval_expr = Py_eval_input,

    eval_single_statement = Py_single_input,

    eval_statements = Py_file_input
};

template <eval_mode start = eval_expr>
object eval(const str &expr, handle global = handle(), handle local = handle()) {
    if (!local.is_valid())
        local = global;

    object codeobj = steal(Py_CompileString(expr.c_str(), "<string>", start));
    if (!codeobj.is_valid())
        raise_python_error();

    PyObject *result = PyEval_EvalCode(codeobj.ptr(), global.ptr(), local.ptr());
    if (!result)
        raise_python_error();

    return steal(result);
}

template <eval_mode start = eval_expr, size_t N>
object eval(const char (&s)[N], handle global = handle(), handle local = handle()) {
    str expr = (s[0] == '\n') ? str(module_::import_("textwrap").attr("dedent")(s)) : str(s);
    return eval<start>(expr, global, local);
}

inline void exec(const str &expr, handle global = handle(), handle local = handle()) {
    eval<eval_statements>(expr, global, local);
}

template <size_t N>
void exec(const char (&s)[N], handle global = handle(), handle local = handle()) {
    eval<eval_statements>(s, global, local);
}

NAMESPACE_END(NB_NAMESPACE)
