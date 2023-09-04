# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from cpython.exc cimport PyErr_SetFromErrno
from cpython.time cimport localtime as _localtime
from libc.time cimport time_t, localtime as libc_localtime

# Struct --------------------------------------------------------------------------------
cdef extern from "<time.h>" nogil:
    cdef struct tm:
        int  tm_sec
        int  tm_min
        int  tm_hour
        int  tm_mday
        int  tm_mon
        int  tm_year
        int  tm_wday
        int  tm_yday
        int  tm_isdst

# Functions -----------------------------------------------------------------------------
# Equivalent to `time.localtime()`.
cdef inline tm localtime() except * nogil:
    return _localtime()

# Equivalent to `time.localtime(timestamp)`.
cdef inline tm localize_time(double timestamp) except *:
    cdef:
        time_t tic = <time_t>timestamp
        tm* tms

    tms = libc_localtime(&tic)
    if tms is NULL:
        raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    tms.tm_year += 1900
    tms.tm_mon += 1
    tms.tm_wday = (tms.tm_wday + 6) % 7
    tms.tm_yday += 1
    return tms[0]

cdef inline int raise_from_errno() except -1 with gil:
    PyErr_SetFromErrno(RuntimeError)
    return <int> -1  # Let the C compiler know that this function always raises.
