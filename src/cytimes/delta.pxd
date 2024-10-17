# cython: language_level=3

from cpython cimport datetime

# Constants
cdef tuple WEEKDAY_REPRS

# Utils
cdef bint is_delta(object o) except -1

# Delta
cdef class Delta:
    cdef:
        int _years
        int _months
        int _days
        int _hours
        int _minutes
        int _seconds
        int _microseconds
        int _year
        int _month
        int _day
        int _weekday
        int _hour
        int _minute
        int _second
        int _microsecond
        long long _hashcode
    # Arithmetic: addition
    cdef inline object _add_date(self, object o)
    cdef inline object _add_datetime(self, object o)
    cdef inline Delta _add_delta(self, Delta o)
    cdef inline Delta _add_timedelta(self, object o)
    cdef inline Delta _add_relativedelta(self, object o)
    cdef inline Delta _add_int(self, int o)
    cdef inline Delta _add_float(self, double o)
    # Arithmetic: right addition
    cdef inline Delta _radd_relativedelta(self, object o)
    # Arithmetic: subtraction
    cdef inline Delta _sub_delta(self, Delta o)
    cdef inline Delta _sub_timedelta(self, object o)
    cdef inline Delta _sub_relativedelta(self, object o)
    cdef inline Delta _sub_int(self, int o)
    cdef inline Delta _sub_float(self, double o)
    # Arithmetic: right subtraction
    cdef inline object _rsub_date(self, object o)
    cdef inline object _rsub_datetime(self, object o)
    cdef inline Delta _rsub_timedelta(self, object o)
    cdef inline Delta _rsub_relativedelta(self, object o)
    cdef inline Delta _rsub_int(self, int o)
    cdef inline Delta _rsub_float(self, double o)
    # Arithmetic: multiplication
    cdef inline Delta _mul_int(self, int i)
    cdef inline Delta _mul_float(self, double f)
    # Comparison
    cdef inline bint _eq_delta(self, Delta o) except -1
    cdef inline bint _eq_timedelta(self, object o) except -1
    cdef inline bint _eq_relativedelta(self, object o) except -1
