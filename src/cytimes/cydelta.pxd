# cython: language_level=3

from cpython cimport datetime

# Constants
cdef:
    # . weekday
    tuple WEEKDAY_REPRS

# Utils
cdef inline int combine_ms_us(int ms, int us) noexcept:
    """Combine millisecond and microsecond into microsecond `<'int'>`.
    Returns -1 if both values are invalid. """
    # Millisecond + Microsecond (fraction)
    cdef int ms_us
    if ms >= 0:
        ms_us = min(ms, 999) * 1_000
        if us > 0:
            ms_us += us % 1_000
        return ms_us
    # Clip Microsecond
    if us >= 0:
        return min(us, 999_999)
    # Invalid Value
    return -1

# cytimedelta
cdef class cytimedelta:
    cdef:
        int _years, _months, _days, _hours
        int _minutes, _seconds, _microseconds
        int _year, _month, _day, _weekday, _hour
        int _minute, _second, _microsecond
        Py_ssize_t _hashcode
    # Special methods: addition
    cdef inline datetime.datetime _add_date(self, datetime.date o)
    cdef inline datetime.datetime _add_datetime(self, datetime.datetime o)
    cdef inline cytimedelta _add_timedelta(self, datetime.timedelta o)
    cdef inline cytimedelta _add_cytimedelta(self, cytimedelta o)
    cdef inline cytimedelta _add_relativedelta(self, object o)
    cdef inline cytimedelta _add_int(self, int o)
    cdef inline cytimedelta _add_float(self, double o)
    cdef inline cytimedelta _radd_relativedelta(self, object o)
    # Special methods: substraction
    cdef inline cytimedelta _sub_timedelta(self, datetime.timedelta o)
    cdef inline cytimedelta _sub_cytimedelta(self, cytimedelta o)
    cdef inline cytimedelta _sub_relativedelta(self, object o)
    cdef inline cytimedelta _sub_int(self, int o)
    cdef inline cytimedelta _sub_float(self, double o)
    cdef inline datetime.datetime _rsub_date(self, datetime.date o)
    cdef inline datetime.datetime _rsub_datetime(self, datetime.datetime o)
    cdef inline cytimedelta _rsub_timedelta(self, datetime.timedelta o)
    cdef inline cytimedelta _rsub_relativedelta(self, object o)
    cdef inline cytimedelta _rsub_int(self, int o)
    cdef inline cytimedelta _rsub_float(self, double o)
    # Special methods: multiplication
    cdef inline cytimedelta _mul_int(self, int factor)
    cdef inline cytimedelta _mul_float(self, double factor)
    # Special methods: comparison
    cdef inline bint _eq_timedelta(self, datetime.timedelta o) except -1
    cdef inline bint _eq_cytimedelta(self, cytimedelta o) except -1
    cdef inline bint _eq_relativedelta(self, object o) except -1
