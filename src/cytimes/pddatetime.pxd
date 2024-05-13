# cython: language_level=3

cimport numpy as np
from numpy cimport PyArray_SETITEM, PyArray_GETPTR1
from cpython cimport datetime
from cytimes.cyparser cimport Config
from cytimes cimport typeref
from cytimes import errors

# Constants
cdef:
    # . unit
    set UNIT_FREQUENCY
    # . function
    object FN_PD_DATERANGE, FN_PD_TODATETIME, FN_PD_TOTIMEDELTA
    object FN_NP_ABS, FN_NP_FULL, FN_NP_WHERE, FN_NP_MIN

# Utils
cdef inline bint is_pddt(object obj) except -1:
    """Check if an object is 'pddt' `<'bool'>`"""
    return isinstance(obj, pddt)

cdef inline object access_pddt(pddt pt):
    """Access the `<'pddt'>` value, returns `<'Series[Timestamp]'>`."""
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
            "<'pddt'>\nInvalid timezone: %s %s." % (type(tz), repr(tz))
        ) from err

cdef inline int ndarray_setitem_1d(np.ndarray arr, np.npy_intp i, object item):
    """Set item for 1-dimensional numpy ndarray `<'int'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    return PyArray_SETITEM(arr, itemptr, item)

# pddt (Pandas Datetime)
cdef class pddt:
    cdef:
        # . value
        object _dt
        # . config
        Config _cfg
        object _day1st, _year1st, _utc, _format, _exact, _default
    # C-API: Access
    cdef inline str _c_unit(self)
    # C-API: Year
    cpdef pddt to_year_1st(self)
    cpdef pddt to_year_lst(self)
    cpdef pddt to_curr_year(self, object month=?, int day=?)
    cpdef pddt to_prev_year(self, object month=?, int day=?)
    cpdef pddt to_next_year(self, object month=?, int day=?)
    cpdef pddt to_year(self, int offset=?, object month=?, int day=?)
    # C-API: Quarter
    cpdef pddt to_quarter_1st(self)
    cpdef pddt to_quarter_lst(self)
    cpdef pddt to_curr_quarter(self, int month=?, int day=?)
    cpdef pddt to_prev_quarter(self, int month=?, int day=?)
    cpdef pddt to_next_quarter(self, int month=?, int day=?)
    cpdef pddt to_quarter(self, int offset=?, int month=?, int day=?)
    # C-API: Month
    cpdef pddt to_month_1st(self)
    cpdef pddt to_month_lst(self)
    cpdef pddt to_curr_month(self, int day=?)
    cpdef pddt to_prev_month(self, int day=?)
    cpdef pddt to_next_month(self, int day=?)
    cpdef pddt to_month(self, int offset=?, int day=?)
    # C-API: Weekday
    cdef inline pddt _c_to_curr_weekday(self, int weekday)
    cpdef pddt to_monday(self)
    cpdef pddt to_tuesday(self)
    cpdef pddt to_wednesday(self)
    cpdef pddt to_thursday(self)
    cpdef pddt to_friday(self)
    cpdef pddt to_saturday(self)
    cpdef pddt to_sunday(self)
    cpdef pddt to_curr_weekday(self, object weekday=?)
    cpdef pddt to_prev_weekday(self, object weekday=?)
    cpdef pddt to_next_weekday(self, object weekday=?)
    cpdef pddt to_weekday(self, int offset=?, object weekday=?)
    # C-API: Day
    cpdef pddt to_yesterday(self)
    cpdef pddt to_tomorrow(self)
    cpdef pddt to_day(self, int offset)
    # C-API: Time
    cpdef pddt to_time_start(self)
    cpdef pddt to_time_end(self)
    cpdef pddt to_time(self, int hour=?, int minute=?, int second=?, int millisecond=?, int microsecond=?)
    # C-API: Timezone
    cdef inline object _c_tz_localize(self, object dt, object tz, object ambiguous, object nonexistent)
    cdef inline object _c_tz_convert(self, object dt, object tz, object ambiguous, object nonexistent)
    cpdef pddt tz_localize(self, object tz=?, object ambiguous=?, object nonexistent=?)
    cpdef pddt tz_convert(self, object tz=?, object ambiguous=?, object nonexistent=?)
    cpdef pddt tz_switch(self, object targ_tz, object base_tz=?, object naive=?, object ambiguous=?, object nonexistent=?)
    cpdef pddt astimezone(self, object tz=?, object ambiguous=?, object nonexistent=?)
    # C-API: Frequency
    cpdef pddt freq_round(self, object freq, object ambiguous=?, object nonexistent=?)
    cpdef pddt freq_ceil(self, object freq, object ambiguous=?, object nonexistent=?)
    cpdef pddt freq_floor(self, object freq, object ambiguous=?, object nonexistent=?)
    # C-API: Delta
    cdef inline object _c_cal_delta(self, object other, object unit, object inclusive)
    cpdef pddt add_delta(
        self, int years=?, int months=?, int days=?, int weeks=?, 
        int hours=?, int minutes=?, int seconds=?, int milliseconds=?, int microseconds=?)
    cpdef pddt sub_delta(
        self, int years=?, int months=?, int days=?, int weeks=?, 
        int hours=?, int minutes=?, int seconds=?, int milliseconds=?, int microseconds=?)
    # C-API: Unit
    cpdef pddt as_unit(self, object unit)
    # Internal methods
    cdef inline pddt _new(self, object dtsobj, object name, object unit, object copy)
    cdef inline datetime.datetime _parse_dtobj(self, object dtobj, object default, bint strict)
    cdef inline object _parse_dtsobj(self, object dtsobj, object default, object name, object unit, bint copy)
    cdef inline object _parse_dtsobj_fallback(self, object dtsobj, object default, object name, object unit, bint is_series, Exception exc)
    cdef inline int _parse_month(self, object month) except -2
    cdef inline int _parse_weekday(self, object weekday) except -2
    cdef inline object _parse_frequency(self, object freq)
    cdef inline object _adjust_unit(self, object dt, object unit, bint copy)
    cdef inline object _arr_to_series(self, object arr)
    cdef inline object _arr_to_tdindex(self, object arr, object unit)
    cdef inline object _obj_to_pddt(self, object obj, object name, object unit)
    cdef inline object _obj_to_series(self, object obj)
    cdef inline bint _validate_amb_nonex(self, object ambiguous, object nonexistent) except -1
    cdef inline bint _raise_error(self, Exception err, str msg=?) except -1
    # Special methods
    cpdef pddt copy(self)
