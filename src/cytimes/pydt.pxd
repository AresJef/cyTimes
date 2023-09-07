# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from cpython cimport datetime
from cytimes cimport cydatetime as cydt
from cytimes.cytimedelta cimport cytimedelta

# Constants
cdef long long US_NULL

# pydt (Python Datetime)
cdef class pydt:
    # Attributes
    cdef:
        object _default, _tzinfos, _parserinfo
        bint _dayfirst, _yearfirst, _ignoretz, _fuzzy
        datetime.datetime _dt
        int __hashcode
        int __year, __month, __day, __fold
        int __hour, __minute, __second, __microsecond
        datetime.tzinfo __tzinfo
        int __quarter, __days_in_month, __weekday
        long long __microseconds
    # Access
    cdef str _dtiso(self) noexcept
    cdef str _dtisotz(self) noexcept
    cdef datetime.date _date(self) noexcept
    cdef str _dateiso(self) noexcept
    cdef datetime.time _time(self) noexcept
    cdef str _timeiso(self) noexcept
    cdef datetime.time _timetz(self) noexcept
    cdef str _timeisotz(self) noexcept
    cdef int _ordinal(self) except -1
    cdef double _seconds(self) noexcept
    cdef double _seconds_utc(self) noexcept
    cdef long long _microseconds(self) noexcept
    cdef long long _microseconds_utc(self) noexcept
    cdef double _timestamp(self) noexcept
    # Absolute
    cdef int _year(self) except -1
    cdef int _month(self) except -1
    cdef int _day(self) except -1
    cdef int _hour(self) except -1
    cdef int _minute(self) except -1
    cdef int _second(self) except -1
    cdef int _microsecond(self) except -1
    cdef datetime.tzinfo _tzinfo(self) noexcept
    cdef int _fold(self) except -1
    # Calendar
    cdef bint _is_leapyear(self) except -1
    cdef int _days_in_year(self) except -1
    cdef int _days_bf_year(self) except -1
    cdef int _days_of_year(self) except -1
    cdef int _quarter(self) except -1
    cdef int _days_in_quarter(self) except -1
    cdef int _days_bf_quarter(self) except -1
    cdef int _days_of_quarter(self) except -1
    cdef int _days_in_month(self) except -1
    cdef int _days_bf_month(self) except -1
    cdef int _weekday(self) except -1
    cdef int _isoweekday(self) except -1
    cdef int _isoweek(self) except -1
    cdef int _isoyear(self) except -1
    cdef cydt.iso _isocalendar(self) except *
    # Time manipulation
    cdef pydt _start_time(self) noexcept
    cdef pydt _end_time(self) noexcept
    # Day manipulation
    cdef pydt _tomorrow(self) noexcept
    cdef pydt _yesterday(self) noexcept
    # Weekday manipulation
    cdef pydt _monday(self) noexcept
    cdef pydt _tuesday(self) noexcept
    cdef pydt _wednesday(self) noexcept
    cdef pydt _thursday(self) noexcept
    cdef pydt _friday(self) noexcept
    cdef pydt _saturday(self) noexcept
    cdef pydt _sunday(self) noexcept
    cdef pydt _curr_week(self, object weekday) except *
    cdef pydt _to_week(self, int offset, object weekday) except *
    cdef bint _is_weekday(self, object weekday) except -1
    # Month manipulation
    cdef pydt _month_1st(self) noexcept
    cdef bint _is_month_1st(self) except -1
    cdef pydt _month_lst(self) noexcept
    cdef bint _is_month_lst(self) except -1
    cdef pydt _curr_month(self, int day) except *
    cdef pydt _to_month(self, int offset, int day) except *
    cdef bint _is_month(self, object month) except -1
    # Quarter manipulation
    cdef pydt _quarter_1st(self) noexcept
    cdef bint _is_quarter_1st(self) except -1
    cdef pydt _quarter_lst(self) noexcept
    cdef bint _is_quarter_lst(self) except -1
    cdef pydt _curr_quarter(self, int month, int day) except *
    cdef pydt _to_quarter(self, int offset, int month, int day) except *
    cdef bint _is_quarter(self, int quarter) except -1
    # Year manipulation
    cdef pydt _year_1st(self) noexcept
    cdef bint _is_year_1st(self) except -1
    cdef pydt _year_lst(self) noexcept
    cdef bint _is_year_lst(self) except -1
    cdef pydt _curr_year(self, object month, int day) except *
    cdef pydt _to_year(self, int offset, object month, int day) except *
    cdef bint _is_year(self, int year) except -1
    # Timezone manipulation
    cdef datetime.datetime _tz_localize(self, datetime.datetime dt, object tz) except *
    cdef datetime.datetime _tz_convert(self, datetime.datetime dt, object tz) except *
    cdef datetime.datetime _tz_switch(self, datetime.datetime dt, object targ_tz, object base_tz, bint naive) except *
    # Frequency manipulation
    cdef pydt _round(self, str freq) except *
    cdef pydt _ceil(self, str freq) except *
    cdef pydt _floor(self, str freq) except *
    # Delta adjustment
    cdef pydt _delta(self, int years, int months, int days, int weeks, int hours, int minutes, int seconds, int microseconds) except *
    # Replace adjustment
    cdef pydt _replace(self, int year, int month, int day, int hour, int minute, int second, int microsecond, object tzinfo, int fold) noexcept
    # Between calculation
    cdef long long _between(self, object obj, str unit, bint inclusive) except *
    # Core methods
    cdef pydt _new(self, datetime.datetime) except *
    cdef datetime.datetime _to_datetime(self, object timeobj) except *
    cdef datetime.datetime _parse_datetime(self, str timestr) except *
    cdef pydt _add_days(self, int days) except *
    cdef int _parse_weekday(self, object weekday) except -1
    cdef int _parse_month(self, object month) except -1
    cdef datetime.tzinfo _parse_tzinfo(self, object tz) except *
    cdef long long _parse_frequency(self, str freq) except -1
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
