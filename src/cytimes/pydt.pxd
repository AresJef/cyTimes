# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from cpython cimport datetime
from cytimes.cytimedelta cimport cytimedelta

# Constants
cdef long long US_NULL

# pydt (Python Datetime)
cdef class pydt:
    cdef:
        object _default, _tzinfos, _parserinfo
        bint _dayfirst, _yearfirst, _ignoretz, _fuzzy
        datetime.datetime _dt
        int _hashcode
        int _year, _month, _day, _hour, _minute, _second, _microsecond
        int _quarter, _month_days, _weekday, _fold
        datetime.tzinfo _tzinfo
        long long _microseconds
    # Core methods
    cdef pydt _new(self, datetime.datetime) except *
    cdef datetime.datetime _to_datetime(self, object timeobj) except *
    cdef int _get_year(self) except -1
    cdef int _get_month(self) except -1
    cdef int _get_day(self) except -1
    cdef int _get_hour(self) except -1
    cdef int _get_minute(self) except -1
    cdef int _get_second(self) except -1
    cdef int _get_microsecond(self) except -1
    cdef int _get_quarter(self) except -1
    cdef int _get_month_days(self) except -1
    cdef int _get_weekday(self) except -1
    cdef int _get_fold(self) except -1
    cdef datetime.tzinfo _get_tzinfo(self) noexcept
    cdef long long _get_microseconds(self) noexcept
    cdef datetime.datetime _parse_datetime(self, str timestr) except *
    cdef pydt _add_days(self, int days) except *
    cdef int _parse_weekday(self, object weekday) except -1
    cdef pydt _curr_week(self, object weekday) except *
    cdef pydt _to_week(self, int offset, object weekday) except *
    cdef int _parse_month(self, object month) except -1
    cdef pydt _curr_month(self, int day) except *
    cdef pydt _to_month(self, int offset, int day) except *
    cdef pydt _curr_quarter(self, int month, int day) except *
    cdef pydt _to_quarter(self, int offset, int month, int day) except *
    cdef pydt _curr_year(self, object month, int day) except *
    cdef pydt _to_year(self, int offset, object month, int day) except *
    cdef datetime.tzinfo _parse_tzinfo(self, object tz) except *
    cdef datetime.datetime _tz_localize(self, datetime.datetime dt, object tz) except *
    cdef datetime.datetime _tz_convert(self, datetime.datetime dt, object tz) except *
    cdef datetime.datetime _tz_switch(self, datetime.datetime dt, object targ_tz, object base_tz, bint naive) except *
    cdef long long _parse_frequency(self, str freq) except -1
    cdef pydt _round_frequency(self, str freq) except *
    cdef pydt _ceil_frequency(self, str freq) except *
    cdef pydt _floor_frequency(self, str freq) except *
    cdef pydt _delta(self, int years, int months, int days, int weeks, int hours, int minutes, int seconds, int microseconds) except *
    cdef pydt _replace(self, int year, int month, int day, int hour, int minute, int second, int microsecond, object tzinfo, int fold) noexcept
    cdef long long _between(self, object obj, str unit, bint inclusive) except *
    cdef long long _between_pydt(self, pydt pt, str unit, bint inclusive) except *
    cdef long long _between_datetime(self, datetime.datetime dt, str unit, bint inclusive) except *
    # Special methods
    cdef pydt _add_timedelta(self, datetime.timedelta other) except *
    cdef pydt _add_cytimedelta(self, cytimedelta other) except *
    cdef pydt _add_relativedelta(self, object other) except *
    cdef datetime.timedelta _sub_pydt(self, pydt other) except *
    cdef datetime.timedelta _sub_datetime(self, datetime.datetime other) except *
    cdef pydt _sub_timedelta(self, datetime.timedelta other) except *
    cdef pydt _sub_cytimedelta(self, cytimedelta other) except *
    cdef pydt _sub_relativedelta(self, object other) except *
    cdef datetime.timedelta _rsub_datetime(self, datetime.datetime other) except *
    cdef int _compare_pydt(self, pydt other, bint allow_mixed) except *
    cdef int _compare_datetime(self, datetime.datetime other, bint allow_mixed) except *
    cdef int _compare_datetime_base(self, datetime.datetime other) except *
    cdef int _compare_datetime_delta(self, datetime.datetime other) except *
    cdef int _hash(self) noexcept
