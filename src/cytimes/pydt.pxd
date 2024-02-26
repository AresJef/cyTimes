# cython: language_level=3

from cpython cimport datetime
from cytimes.cyparser cimport Config
from cytimes cimport cydatetime as cydt
from cytimes.cytimedelta cimport cytimedelta

# Constants
# . timezone
cdef:
    object RLDELTA_DTYPE, ZONEINFO_DTYPE
    set TIMEZONES_AVAILABLE

# pydt (Python Datetime)
cdef datetime.datetime access_pydt_datetime(pydt pt) noexcept
cdef object parse_tzinfo(object tz) except *

cdef class pydt:
    cdef:
        # . config
        Config _cfg
        object _default
        bint _day1st, _year1st, _ignoretz, _fuzzy
        # . datetime
        datetime.datetime _dt
        # . hashcode
        int _hashcode
    # Access
    cdef unsigned int _capi_year(pydt self) except -1
    cdef unsigned int _capi_quarter(pydt self) except -1
    cdef unsigned int _capi_month(pydt self) except -1
    cdef unsigned int _capi_day(pydt self) except -1
    cdef unsigned int _capi_hour(pydt self) except -1
    cdef unsigned int _capi_minute(pydt self) except -1
    cdef unsigned int _capi_second(pydt self) except -1
    cdef unsigned int _capi_microsecond(pydt self) except -1
    cdef datetime.tzinfo _capi_tzinfo(pydt self) noexcept
    cdef unsigned int _capi_fold(pydt self) except -1
    cdef datetime.datetime _capi_dt(pydt self) noexcept
    cdef str _capi_dt_str(pydt self) noexcept
    cdef str _capi_dt_iso(pydt self) noexcept
    cdef str _capi_dt_isotz(pydt self) noexcept
    cdef datetime.date _capi_date(pydt self) noexcept
    cdef str _capi_date_iso(pydt self) noexcept
    cdef datetime.time _capi_time(pydt self) noexcept
    cdef datetime.time _capi_timetz(pydt self) noexcept
    cdef str _capi_time_iso(pydt self) noexcept
    cdef unsigned int _capi_ordinal(pydt self) except -1
    cdef double _capi_seconds(pydt self) noexcept
    cdef double _capi_seconds_utc(pydt self) noexcept
    cdef long long _capi_microseconds(pydt self) noexcept
    cdef long long _capi_microseconds_utc(pydt self) noexcept
    cdef double _capi_timestamp(pydt self) except *
    # Calendar: Year
    cdef bint _capi_is_leapyear(pydt self) except -1
    cdef unsigned int _capi_leap_bt_years(pydt self, unsigned int year) except -1
    cdef unsigned int _capi_days_in_year(pydt self) except -1
    cdef unsigned int _capi_days_bf_year(pydt self) except -1
    cdef unsigned int _capi_days_of_year(pydt self) except -1
    # Manipulate: Year
    cdef bint _capi_is_year(pydt self, int year) except -1
    cdef bint _capi_is_year_1st(pydt self) except -1
    cdef bint _capi_is_year_lst(pydt self) except -1
    cdef pydt _capi_to_year_1st(pydt self) noexcept
    cdef pydt _capi_to_year_lst(pydt self) noexcept
    cdef pydt _capi_to_curr_year(pydt self, object month, int day) except *
    cdef pydt _capi_to_year(pydt self, int offset, object month, int day) except *
    # Calendar: Quarter
    cdef unsigned int _capi_days_in_quarter(pydt self) except -1
    cdef unsigned int _capi_days_bf_quarter(pydt self) except -1
    cdef unsigned int _capi_days_of_quarter(pydt self) except -1
    cdef unsigned int _capi_quarter_1st_month(pydt self) except -1
    cdef unsigned int _capi_quarter_lst_month(pydt self) except -1
    # Manipulate: Quarter
    cdef bint _capi_is_quarter(pydt self, int quarter) except -1
    cdef bint _capi_is_quarter_1st(pydt self) except -1
    cdef bint _capi_is_quarter_lst(pydt self) except -1
    cdef pydt _capi_to_quarter_1st(pydt self) noexcept
    cdef pydt _capi_to_quarter_lst(pydt self) noexcept
    cdef pydt _capi_to_curr_quarter(pydt self, int month, int day) noexcept
    cdef pydt _capi_to_quarter(pydt self, int offset, int month, int day) noexcept
    # Calendar: Month
    cdef unsigned int _capi_days_in_month(pydt self) except -1
    cdef unsigned int _capi_days_bf_month(pydt self) except -1
    # Manipulate: Month
    cdef bint _capi_is_month(pydt self, object month) except -1
    cdef bint _capi_is_month_1st(pydt self) except -1
    cdef bint _capi_is_month_lst(pydt self) except -1
    cdef pydt _capi_to_month_1st(pydt self) noexcept
    cdef pydt _capi_to_month_lst(pydt self) noexcept
    cdef pydt _capi_to_curr_month(pydt self, int day) noexcept
    cdef pydt _capi_to_month(pydt self, int offset, int day) noexcept
    # Calendar: Weekday
    cdef unsigned int _capi_weekday(pydt self) except -1
    cdef unsigned int _capi_isoweekday(pydt self) except -1
    cdef unsigned int _capi_isoweek(pydt self) except -1
    cdef unsigned int _capi_isoyear(pydt self) except -1
    cdef cydt.iso _capi_isocalendar(pydt self) noexcept
    # Manipulate: Weekday
    cdef bint _capi_is_weekday(pydt self, object weekday) except -1
    cdef pydt _capi_to_curr_weekday_int(pydt self, unsigned int weekday) noexcept
    cdef pydt _capi_to_curr_weekday(pydt self, object weekday) except *
    cdef pydt _capi_to_weekday(pydt self, int offset, object weekday) except *
    # Manipulate: Day
    cdef bint _capi_is_day(pydt self, int day) except -1
    cdef pydt _capi_to_day(pydt self, int offset) noexcept
    # Manipulate: Time
    cdef bint _capi_is_time_start(pydt self) except -1
    cdef bint _capi_is_time_end(pydt self) except -1
    cdef pydt _capi_to_time_start(pydt self) noexcept
    cdef pydt _capi_to_time_end(pydt self) noexcept
    cdef pydt _capi_to_time(pydt self, int hour, int minute, int second, int microsecond) noexcept
    # Manipulate: Timezone
    cdef set _capi_tz_available(pydt self) noexcept
    cdef pydt _capi_tz_localize(pydt self, object tz) except *
    cdef pydt _capi_tz_convert(pydt self, object tz) except *
    cdef pydt _capi_tz_switch(pydt self, object targ_tz, object base_tz=?, bint naive=?) except *
    # Manipulate: Frequency
    cdef pydt _capi_freq_round(pydt self, object freq) except *
    cdef pydt _capi_freq_ceil(pydt self, object freq) except *
    cdef pydt _capi_freq_floor(pydt self, object freq) except *
    # Manipulate: Delta
    cdef long long _capi_cal_delta(pydt self, object other, object unit, bint inclusive=?) except *
    # Manipulate: Replace
    cdef pydt _capi_replace(
        pydt self, int year=?, int month=?, int day=?, int hour=?, int minute=?, 
        int second=?, int microsecond=?, object tzinfo=?, int fold=?) except *
    # Core methods
    cdef pydt _new(pydt self, datetime.datetime dt) noexcept
    cdef datetime.datetime _parse_dtobj(pydt self, object dtobj) except *
    cdef datetime.datetime _parse_dtstr(pydt self, object dtstr) except *
    cdef unsigned int _parse_month(pydt self, object month) except -1
    cdef unsigned int _parse_weekday(pydt self, object weekday) except -1
    cdef object _parse_tzinfo(pydt self, object tz) except *
    cdef long long _parse_frequency(pydt self, object freq) except -1
    # Special methods: addition
    cdef pydt _add_timedelta(pydt self, datetime.timedelta delta) except *
    cdef pydt _add_cytimedelta(pydt self, cytimedelta delta) except *
    cdef pydt _add_relativedelta(pydt self, object delta) except *
    # Special methods: substraction
    cdef pydt _sub_timedelta(pydt self, datetime.timedelta delta) except *
    cdef pydt _sub_cytimedelta(pydt self, cytimedelta delta) except *
    cdef pydt _sub_relativedelta(pydt self, object delta) except *
    cdef datetime.timedelta _sub_datetime(pydt self, datetime.datetime dt) except *
    cdef datetime.timedelta _rsub_datetime(pydt self, datetime.datetime dt) except *
    # Special methods: hash
    cdef int _hash(pydt self) except -1
