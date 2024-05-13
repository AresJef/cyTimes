# cython: language_level=3

from cpython cimport datetime
from cytimes.cyparser cimport Config
from cytimes cimport cydatetime as cydt, typeref
from cytimes import errors

# Constants
cdef:
    # . timezones
    set TIMEZONE_AVAILABLE

# Utils
cdef inline bint is_pydt(object obj) except -1:
    """Check if an object is 'pydt' `<'bool'>`"""
    return isinstance(obj, pydt)

cdef inline datetime.datetime access_pydt(pydt pt):
    """Access the `<'pydt'>` value, returns `<'datetime.datetime'>`."""
    return pt._dt

cdef inline object parse_tzinfo(object tz):
    """Parse 'tz' into `<datetime.tzinfo>`. Accept both `<'str'>` 
    (timezone name) and `<'tzinfo'>` instance."""
    if tz is None or datetime.PyTZInfo_Check(tz):
        return tz
    try:
        return typeref.ZONEINFO(tz)
    except Exception as err:
        raise errors.InvalidTimezoneError(
            "<'pydt'>\nInvalid timezone: %s %s." % (type(tz), repr(tz))
        ) from err

# pydt (Python Datetime)
cdef class pydt:
    cdef:
        # . value
        datetime.datetime _dt
        # . config
        Config _cfg
        object _day1st, _year1st, _ignoretz, _isoformat, _default
        # . hashcode
        Py_ssize_t _hashcode
    # C-API: Access
    cdef inline str _c_tzname(self)
    cdef inline datetime.timedelta _c_dst(self)
    cdef inline datetime.timedelta _c_utcoffset(self)
    cdef inline int _c_quarter(self) except -1
    cdef inline str _c_dt_str(self)
    cdef inline str _c_dt_strtz(self)
    cdef inline str _c_dt_iso(self)
    cdef inline str _c_dt_isotz(self)
    cdef inline datetime.date _c_date(self)
    cdef inline str _c_date_iso(self) 
    cdef inline datetime.time _c_time(self)
    cdef inline datetime.time _c_timetz(self)
    cdef inline str _c_time_iso(self)
    cdef inline cydt.tm _c_stime(self) except *
    cdef inline cydt.tm _c_stime_utc(self) except *
    cdef inline int _c_ordinal(self) except -1
    cdef inline double _c_seconds(self) noexcept
    cdef inline double _c_seconds_utc(self) noexcept
    cdef inline long long _c_microseconds(self) noexcept
    cdef inline long long _c_microseconds_utc(self) noexcept
    cdef inline double _c_timestamp(self) except *
    # C-API: Year
    cdef inline bint _c_is_leapyear(self) except -1
    cdef inline int _c_leap_bt_years(self, int year) except -1
    cdef inline int _c_days_in_year(self) except -1
    cdef inline int _c_days_bf_year(self) except -1
    cdef inline int _c_days_of_year(self) except -1
    cdef inline bint _c_is_year(self, int year) except -1
    cdef inline bint _c_is_year_1st(self) except -1
    cdef inline bint _c_is_year_lst(self) except -1
    cpdef pydt to_year_1st(self)
    cpdef pydt to_year_lst(self)
    cpdef pydt to_curr_year(self, object month=?, int day=?)
    cpdef pydt to_prev_year(self, object month=?, int day=?)
    cpdef pydt to_next_year(self, object month=?, int day=?)
    cpdef pydt to_year(self, int offset=?, object month=?, int day=?)
    # C-API: Quater
    cdef inline int _c_days_in_quarter(self) except -1
    cdef inline int _c_days_bf_quarter(self) except -1
    cdef inline int _c_days_of_quarter(self) except -1
    cdef inline int _c_quarter_1st_month(self) except -1
    cdef inline int _c_quarter_lst_month(self) except -1
    cdef inline bint _c_is_quarter(self, int quarter) except -1
    cdef inline bint _c_is_quarter_1st(self) except -1
    cdef inline bint _c_is_quarter_lst(self) except -1
    cpdef pydt to_quarter_1st(self)
    cpdef pydt to_quarter_lst(self)
    cpdef pydt to_curr_quarter(self, int month=?, int day=?)
    cpdef pydt to_prev_quarter(self, int month=?, int day=?)
    cpdef pydt to_next_quarter(self, int month=?, int day=?)
    cpdef pydt to_quarter(self, int offset=?, int month=?, int day=?)
    # C-API: Month
    cdef inline int _c_days_in_month(self) except -1
    cdef inline int _c_days_bf_month(self) except -1
    cdef inline bint _c_is_month(self, object month) except -1
    cdef inline bint _c_is_month_1st(self) except -1
    cdef inline bint _c_is_month_lst(self) except -1
    cpdef pydt to_month_1st(self)
    cpdef pydt to_month_lst(self)
    cpdef pydt to_curr_month(self, int day=?)
    cpdef pydt to_prev_month(self, int day=?)
    cpdef pydt to_next_month(self, int day=?)
    cpdef pydt to_month(self, int offset=?, int day=?)
    # C-API: Weekday
    cdef inline int _c_weekday(self) except -1
    cdef inline int _c_isoweekday(self) except -1
    cdef inline int _c_isoweek(self) except -1
    cdef inline int _c_isoyear(self) except -1
    cdef inline cydt.iso _c_isocalendar(self) noexcept
    cdef inline bint _c_is_weekday(self, object weekday) except -1
    cdef inline pydt _c_to_curr_weekday(self, int weekday)
    cpdef pydt to_monday(self)
    cpdef pydt to_tuesday(self)
    cpdef pydt to_wednesday(self)
    cpdef pydt to_thursday(self)
    cpdef pydt to_friday(self)
    cpdef pydt to_saturday(self)
    cpdef pydt to_sunday(self)
    cpdef pydt to_curr_weekday(self, object weekday=?)
    cpdef pydt to_prev_weekday(self, object weekday=?)
    cpdef pydt to_next_weekday(self, object weekday=?)
    cpdef pydt to_weekday(self, int offset=?, object weekday=?)
    # C-API: Day
    cdef inline bint _c_is_day(self, int day) except -1
    cpdef pydt to_yesterday(self)
    cpdef pydt to_tomorrow(self)
    cpdef pydt to_day(self, int offset=?)
    # C-API: Time
    cdef inline bint _c_is_time_start(self) except -1
    cdef inline bint _c_is_time_end(self) except -1
    cpdef pydt to_time_start(self)
    cpdef pydt to_time_end(self)
    cpdef pydt to_time(
        self, int hour=?, int minute=?, int 
        second=?, int millisecond=?, int microsecond=?)
    # C-API: Timezone
    cpdef pydt tz_localize(self, object tz=?)
    cpdef pydt tz_convert(self, object tz=?)
    cpdef pydt tz_switch(self, object targ_tz, object base_tz=?, object naive=?)
    cpdef pydt astimezone(self, object tz=?)
    # C-API: Frequency
    cpdef pydt freq_round(self, object freq)
    cpdef pydt freq_ceil(self, object freq)
    cpdef pydt freq_floor(self, object freq)
    # C-API: Delta
    cdef inline long long _c_cal_delta(self, object other, object unit, object inclusive) except -2
    cpdef pydt add_delta(
        self,  int years=?, int months=?, int days=?, int weeks=?, 
        int hours=?, int minutes=?, int seconds=?, int milliseconds=?, int microseconds=?)
    cpdef pydt sub_delta(
        self,  int years=?, int months=?, int days=?, int weeks=?, 
        int hours=?, int minutes=?, int seconds=?, int milliseconds=?, int microseconds=?)
    # Internal methods
    cdef inline pydt _new(self, object dtobj)
    cdef inline datetime.datetime _parse_dtobj(self, object dtobj, object default)
    cdef inline int _parse_month(self, object month) except -2
    cdef inline int _parse_weekday(self, object weekday) except -2
    cdef inline long long _parse_frequency(self, object freq) except -1
