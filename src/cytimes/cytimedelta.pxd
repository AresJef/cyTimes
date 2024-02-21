# cython: language_level=3

from cpython cimport datetime

# Constants
cdef:
    object RLDELTA_DTYPE
    tuple WEEKDAY_REPRS

# cytimedelta
cdef class cytimedelta:
    cdef:
        long long _years, _months, _days, _hours
        long long _minutes, _seconds, _microseconds
        int _year, _month, _day, _weekday, _hour
        int _minute, _second, _microsecond
        long long _hashcode
    # Special methods: addition
    cdef datetime.datetime _add_date(cytimedelta self, datetime.date o) except *
    cdef datetime.datetime _add_datetime(cytimedelta self, datetime.datetime o) except *
    cdef cytimedelta _add_cytimedelta(cytimedelta self, cytimedelta o) except *
    cdef cytimedelta _add_timedelta(cytimedelta self, datetime.timedelta o) except *
    cdef cytimedelta _add_relativedelta(cytimedelta self, object o) except *
    cdef cytimedelta _radd_relativedelta(cytimedelta self, object o) except *
    # Special methods: substraction
    cdef cytimedelta _sub_cytimedelta(cytimedelta self, cytimedelta o) except *
    cdef cytimedelta _sub_timedelta(cytimedelta self, datetime.timedelta o) except *
    cdef cytimedelta _sub_relativedelta(cytimedelta self, object o) except *
    cdef datetime.datetime _rsub_date(cytimedelta self, datetime.date o) except *
    cdef datetime.datetime _rsub_datetime(cytimedelta self, datetime.datetime o) except *
    cdef cytimedelta _rsub_timedelta(cytimedelta self, datetime.timedelta o) except *
    cdef cytimedelta _rsub_relativedelta(cytimedelta self, object o) except *
    # Special methods: multiplication
    cdef cytimedelta _mul_int(cytimedelta self, int factor) except *
    cdef cytimedelta _mul_float(cytimedelta self, double factor) except *
    # Special methods: comparison
    cdef bint _eq_cytimedelta(cytimedelta self, cytimedelta o) except -1
    cdef bint _eq_relativedelta(cytimedelta self, object o) except -1
