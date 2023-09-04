# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Struct
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
        # GNU specific extensions
        char *tm_zone
        long tm_gmtoff

# Functions
cdef tm localtime() except * nogil
cdef tm localize_time(double timestamp) except *
