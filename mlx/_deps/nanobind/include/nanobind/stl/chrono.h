#pragma once

#include <nanobind/nanobind.h>

#if !defined(__STDC_WANT_LIB_EXT1__)
#define __STDC_WANT_LIB_EXT1__ 1 // for localtime_s
#endif
#include <time.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>

#include <nanobind/stl/detail/chrono.h>

template <typename type> class duration_caster {
public:
    using rep = typename type::rep;
    using period = typename type::period;
    using duration_t = std::chrono::duration<rep, period>;

    bool from_python(handle src, uint8_t /*flags*/, cleanup_list*) noexcept {
        namespace ch = std::chrono;

        if (!src) return false;

        using days = ch::duration<int_least32_t, std::ratio<86400>>;

        int dd, ss, uu;
        try {
            if (unpack_timedelta(src.ptr(), &dd, &ss, &uu)) {
                value = type(ch::duration_cast<duration_t>(
                                 days(dd) + ch::seconds(ss) + ch::microseconds(uu)));
                return true;
            }
        } catch (python_error& e) {
            e.discard_as_unraisable(src.ptr());
            return false;
        }

        int is_float;
#if defined(Py_LIMITED_API)
        is_float = PyType_IsSubtype(Py_TYPE(src.ptr()), &PyFloat_Type);
#else
        is_float = PyFloat_Check(src.ptr());
#endif
        if (is_float) {
            value = type(ch::duration_cast<duration_t>(
                             ch::duration<double>(PyFloat_AsDouble(src.ptr()))));
            return true;
        }
        return false;
    }

    static const duration_t& get_duration(const duration_t& src) {
        return src;
    }

    template <typename Clock>
    static duration_t get_duration(
            const std::chrono::time_point<Clock, duration_t>& src) {
        return src.time_since_epoch();
    }

    static handle from_cpp(const type& src, rv_policy, cleanup_list*) noexcept {
        namespace ch = std::chrono;

        auto d = get_duration(src);

        using dd_t = ch::duration<int, std::ratio<86400>>;
        using ss_t = ch::duration<int, std::ratio<1>>;
        using us_t = ch::duration<int, std::micro>;

        auto dd = ch::duration_cast<dd_t>(d);
        auto subd = d - dd;
        auto ss = ch::duration_cast<ss_t>(subd);
        auto us = ch::duration_cast<us_t>(subd - ss);
        return pack_timedelta(dd.count(), ss.count(), us.count());
    }

    NB_TYPE_CASTER(type, const_name("datetime.timedelta"))
};

template <class... Args>
auto can_localtime_s(Args*... args) ->
    decltype((localtime_s(args...), std::true_type{}));
std::false_type can_localtime_s(...);

template <class... Args>
auto can_localtime_r(Args*... args) ->
    decltype((localtime_r(args...), std::true_type{}));
std::false_type can_localtime_r(...);

template <class Time, class Buf>
inline std::tm *localtime_thread_safe(const Time *time, Buf *buf) {
    if constexpr (decltype(can_localtime_s(time, buf))::value) {
        std::tm* ret = localtime_s(time, buf);
        return ret;
    } else if constexpr (decltype(can_localtime_s(buf, time))::value) {
        int ret = localtime_s(buf, time);
        return ret == 0 ? buf : nullptr;
    } else {
        static_assert(decltype(can_localtime_r(time, buf))::value,
                      "<nanobind/stl/chrono.h> type caster requires "
                      "that your C library support localtime_r or localtime_s");
        std::tm* ret = localtime_r(time, buf);
        return ret;
    }
}

template <typename Duration>
class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
    using type = std::chrono::time_point<std::chrono::system_clock, Duration>;
    bool from_python(handle src, uint8_t /*flags*/, cleanup_list*) noexcept {
        namespace ch = std::chrono;

        if (!src)
            return false;

        std::tm cal;
        ch::microseconds msecs;
        int yy, mon, dd, hh, min, ss, uu;
        try {
            if (!unpack_datetime(src.ptr(), &yy, &mon, &dd,
                                 &hh, &min, &ss, &uu)) {
                return false;
            }
        } catch (python_error& e) {
            e.discard_as_unraisable(src.ptr());
            return false;
        }
        cal.tm_sec = ss;
        cal.tm_min = min;
        cal.tm_hour = hh;
        cal.tm_mday = dd;
        cal.tm_mon = mon - 1;
        cal.tm_year = yy - 1900;
        cal.tm_isdst = -1;
        msecs = ch::microseconds(uu);
        value = ch::time_point_cast<Duration>(
                ch::system_clock::from_time_t(std::mktime(&cal)) + msecs);
        return true;
    }

    static handle from_cpp(const type& src, rv_policy, cleanup_list*) noexcept {
        namespace ch = std::chrono;

        using us_t = ch::duration<std::int64_t, std::micro>;
        auto us = ch::duration_cast<us_t>(src.time_since_epoch() %
                                          ch::seconds(1));
        if (us.count() < 0)
            us += ch::seconds(1);

        std::time_t tt = ch::system_clock::to_time_t(
                ch::time_point_cast<ch::system_clock::duration>(src - us));

        std::tm localtime;
        if (!localtime_thread_safe(&tt, &localtime)) {
            PyErr_Format(PyExc_ValueError,
                         "Unable to represent system_clock in local time; "
                         "got time_t %ld", static_cast<std::int64_t>(tt));
            return handle();
        }
        return pack_datetime(localtime.tm_year + 1900,
                             localtime.tm_mon + 1,
                             localtime.tm_mday,
                             localtime.tm_hour,
                             localtime.tm_min,
                             localtime.tm_sec,
                             (int) us.count());
    }
    NB_TYPE_CASTER(type, const_name("datetime.datetime"))
};

template <typename Clock, typename Duration>
class type_caster<std::chrono::time_point<Clock, Duration>>
  : public duration_caster<std::chrono::time_point<Clock, Duration>> {};

template <typename Rep, typename Period>
class type_caster<std::chrono::duration<Rep, Period>>
  : public duration_caster<std::chrono::duration<Rep, Period>> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
