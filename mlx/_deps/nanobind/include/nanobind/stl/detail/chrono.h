#pragma once

#include <nanobind/nanobind.h>

#if !defined(Py_LIMITED_API) && !defined(PYPY_VERSION)
#  include <datetime.h>
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wattributes"
#endif

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

bool unpack_timedelta(PyObject *o, int *days, int *secs, int *usecs);

bool unpack_datetime(PyObject *o, int *year, int *month, int *day,
                     int *hour, int *minute, int *second,
                     int *usec);

PyObject* pack_timedelta(int days, int secs, int usecs) noexcept;

PyObject* pack_datetime(int year, int month, int day,
                        int hour, int minute, int second,
                        int usec) noexcept;

#if defined(Py_LIMITED_API) || defined(PYPY_VERSION)

struct datetime_types_t {
    handle datetime;
    handle time;
    handle date;
    handle timedelta;

    void ensure_ready() {
        if (datetime.is_valid())
            return;

        object mod = module_::import_("datetime");
        object datetime_o = mod.attr("datetime");
        object time_o = mod.attr("time");
        object date_o = mod.attr("date");
        object timedelta_o = mod.attr("timedelta");

        datetime = datetime_o.release();
        time = time_o.release();
        date = date_o.release();
        timedelta = timedelta_o.release();
    }
};

inline datetime_types_t datetime_types;

NB_NOINLINE inline bool set_from_int_attr(int *dest, PyObject *o,
                                          const char *name) noexcept {
    PyObject *value = PyObject_GetAttrString(o, name);
    if (!value)
        return false;
    long lval = PyLong_AsLong(value);
    if (lval == -1 && PyErr_Occurred()) {
        Py_DECREF(value);
        return false;
    }
    if (lval < std::numeric_limits<int>::min() ||
        lval > std::numeric_limits<int>::max()) {
        PyErr_Format(PyExc_OverflowError,
                     "%R attribute '%s' (%R) does not fit in an int",
                     o, name, value);
        Py_DECREF(value);
        return false;
    }
    Py_DECREF(value);
    *dest = static_cast<int>(lval);
    return true;
}

NB_NOINLINE inline bool unpack_timedelta(PyObject *o, int *days,
                                         int *secs, int *usecs) {
    datetime_types.ensure_ready();
    if (PyType_IsSubtype(Py_TYPE(o),
                         (PyTypeObject *) datetime_types.timedelta.ptr())) {
        if (!set_from_int_attr(days, o, "days") ||
            !set_from_int_attr(secs, o, "seconds") ||
            !set_from_int_attr(usecs, o, "microseconds")) {
            raise_python_error();
        }
        return true;
    }
    return false;
}

NB_NOINLINE inline bool unpack_datetime(PyObject *o,
                                        int *year, int *month, int *day,
                                        int *hour, int *minute, int *second,
                                        int *usec) {
    datetime_types.ensure_ready();
    if (PyType_IsSubtype(Py_TYPE(o),
                         (PyTypeObject *) datetime_types.datetime.ptr())) {
        if (!set_from_int_attr(usec, o, "microsecond") ||
            !set_from_int_attr(second, o, "second") ||
            !set_from_int_attr(minute, o, "minute") ||
            !set_from_int_attr(hour, o, "hour") ||
            !set_from_int_attr(day, o, "day") ||
            !set_from_int_attr(month, o, "month") ||
            !set_from_int_attr(year, o, "year")) {
            raise_python_error();
        }
        return true;
    }
    if (PyType_IsSubtype(Py_TYPE(o),
                         (PyTypeObject *) datetime_types.date.ptr())) {
        *usec = *second = *minute = *hour = 0;
        if (!set_from_int_attr(day, o, "day") ||
            !set_from_int_attr(month, o, "month") ||
            !set_from_int_attr(year, o, "year")) {
            raise_python_error();
        }
        return true;
    }
    if (PyType_IsSubtype(Py_TYPE(o),
                         (PyTypeObject *) datetime_types.time.ptr())) {
        *day = 1;
        *month = 1;
        *year = 1970;
        if (!set_from_int_attr(usec, o, "microsecond") ||
            !set_from_int_attr(second, o, "second") ||
            !set_from_int_attr(minute, o, "minute") ||
            !set_from_int_attr(hour, o, "hour")) {
            raise_python_error();
        }
        return true;
    }
    return false;
}

inline PyObject* pack_timedelta(int days, int secs, int usecs) noexcept {
    try {
        datetime_types.ensure_ready();
        return datetime_types.timedelta(days, secs, usecs).release().ptr();
    } catch (python_error& e) {
        e.restore();
        return nullptr;
    }
}

inline PyObject* pack_datetime(int year, int month, int day,
                               int hour, int minute, int second,
                               int usec) noexcept {
    try {
        datetime_types.ensure_ready();
        return datetime_types.datetime(
                year, month, day, hour, minute, second, usec).release().ptr();
    } catch (python_error& e) {
        e.restore();
        return nullptr;
    }
}

#else // !defined(Py_LIMITED_API) && !defined(PYPY_VERSION)

NB_NOINLINE inline bool unpack_timedelta(PyObject *o, int *days,
                                         int *secs, int *usecs) {
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
        if (!PyDateTimeAPI)
            raise_python_error();
    }
    if (PyDelta_Check(o)) {
        *days = PyDateTime_DELTA_GET_DAYS(o);
        *secs = PyDateTime_DELTA_GET_SECONDS(o);
        *usecs = PyDateTime_DELTA_GET_MICROSECONDS(o);
        return true;
    }
    return false;
}

NB_NOINLINE inline bool unpack_datetime(PyObject *o,
                                        int *year, int *month, int *day,
                                        int *hour, int *minute, int *second,
                                        int *usec) {
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
        if (!PyDateTimeAPI)
            raise_python_error();
    }
    if (PyDateTime_Check(o)) {
        *usec = PyDateTime_DATE_GET_MICROSECOND(o);
        *second = PyDateTime_DATE_GET_SECOND(o);
        *minute = PyDateTime_DATE_GET_MINUTE(o);
        *hour = PyDateTime_DATE_GET_HOUR(o);
        *day = PyDateTime_GET_DAY(o);
        *month = PyDateTime_GET_MONTH(o);
        *year = PyDateTime_GET_YEAR(o);
        return true;
    }
    if (PyDate_Check(o)) {
        *usec = 0;
        *second = 0;
        *minute = 0;
        *hour = 0;
        *day = PyDateTime_GET_DAY(o);
        *month = PyDateTime_GET_MONTH(o);
        *year = PyDateTime_GET_YEAR(o);
        return true;
    }
    if (PyTime_Check(o)) {
        *usec = PyDateTime_TIME_GET_MICROSECOND(o);
        *second = PyDateTime_TIME_GET_SECOND(o);
        *minute = PyDateTime_TIME_GET_MINUTE(o);
        *hour = PyDateTime_TIME_GET_HOUR(o);
        *day = 1;
        *month = 1;
        *year = 1970;
        return true;
    }
    return false;
}

inline PyObject* pack_timedelta(int days, int secs, int usecs) noexcept {
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
        if (!PyDateTimeAPI)
            return nullptr;
    }
    return PyDelta_FromDSU(days, secs, usecs);
}

inline PyObject* pack_datetime(int year, int month, int day,
                               int hour, int minute, int second,
                               int usec) noexcept {
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
        if (!PyDateTimeAPI)
            return nullptr;
    }
    return PyDateTime_FromDateAndTime(year, month, day,
                                      hour, minute, second, usec);
}

#endif
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
