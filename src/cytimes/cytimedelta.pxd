# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from cpython cimport datetime

# Weekday
cdef class Weekday:
    cdef:
        str _weekcode
        int _weekday, _week_offset
    # Special methods
    cdef bint _equal_weekday(self, Weekday other) except -1
    cdef bint _equal_relweekday(self, object other) except -1

cdef Weekday WEEKDAY_NULL

# Cytimedelta
cdef class cytimedelta:
    cdef: 
        int _years, _months, _days, _leapdays
        long long _hours, _minutes, _seconds, _microseconds
        int _year, _month, _day, _hour, _minute, _second, _microsecond
        Weekday _weekday
    # Convert relativedelta.weekday
    cdef Weekday _convert_relweekday(self, object weekday) except *
    # Adjustments
    cdef _adjust_absolute(self, int yearday, int nlyearday) noexcept
    cdef _adjust_relative(self) noexcept
    # Relative information
    cdef int _cal_weeks(self) noexcept
    # Special methods - Addition
    cdef Weekday _pref_relativedelta_weekday(self, object other) except *
    cdef datetime.datetime _add_date_time(self, datetime.date other) noexcept
    cdef cytimedelta _add_cytimedelta(self, cytimedelta other) noexcept
    cdef cytimedelta _add_relativedelta(self, object other) except *
    cdef cytimedelta _add_timedelta(self, datetime.timedelta other) noexcept
    cdef cytimedelta _radd_relativedelta(self, object other) except *
    # Special methods - Substraction
    cdef Weekday _pref_cytimedelta_weekday(self, object other) except *
    cdef cytimedelta _sub_cytimedelta(self, cytimedelta other) noexcept
    cdef cytimedelta _sub_relativedelta(self, object other) except *
    cdef cytimedelta _sub_timedelta(self, datetime.timedelta other) noexcept
    cdef datetime.datetime _rsub_date_time(self, datetime.date other) noexcept
    cdef cytimedelta _rsub_relativedelta(self, object other) except *
    cdef cytimedelta _rsub_timedelta(self, datetime.timedelta other) noexcept
    # Special methods - Multiplication
    cdef cytimedelta _multiply(self, double factor) noexcept
    # Special methods - Manipulation
    cdef cytimedelta _negate(self) noexcept
    cdef cytimedelta _absolute(self) noexcept
    # Special methods
    cdef bint _equal_cytimedelta(self, cytimedelta other) except -1
    cdef bint _equal_relativedelta(self, object other) except -1
    cdef str _represent(self) noexcept
