# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

cimport numpy as np
from cpython cimport datetime

# Constants
cdef np.ndarray DAYS_BR_QUARTER, FIXED_FREQUENCY

# pddt (Pandas Datetime)
cdef class pddt:
    # Attributes
    cdef: 
        datetime.datetime _default
        bint _dayfirst, _yearfirst, _utc, _exact
        str _format
        object _series, _index, _naive
        object __year, __year_1st, __year_lst, __is_leapyear, __days_of_year
        object __quarter, __quarter_1st, __quarter_lst
        object __month, __month_1st, __month_lst, __days_in_month
        object __day, __weekday
    # Absolute
    cdef object _year(self) noexcept
    cdef object _month(self) noexcept
    cdef object _day(self) noexcept
    # Calendar
    cdef object _is_leapyear(self) noexcept
    cdef object _days_of_year(self) noexcept
    cdef object _quarter(self) noexcept
    cdef object _days_in_month(self) noexcept
    cdef object _weekday(self) noexcept
    # Weekday manipulation
    cdef pddt _curr_week(self, object weekday) except *
    cdef pddt _to_week(self, int offset, object weekday) except *
    # Month manipulation
    cdef object _month_1st(self) except *
    cdef object _month_lst(self) except *
    cdef pddt _curr_month(self, int day) except *
    cdef pddt _to_month(self, int offset, int day) except *
    # Quarter manipulation
    cdef object _quarter_1st(self) except *
    cdef object _quarter_lst(self) except *
    cdef pddt _curr_quarter(self, int month, int day) except *
    cdef pddt _to_quarter(self, int offset, int month, int day) except *
    # Year manipulation
    cdef object _year_1st(self) except *
    cdef object _year_lst(self) except *
    cdef pddt _curr_year(self, object month, int day) except *
    cdef pddt _to_year(self, int offset, object month, int day) except *
    # Timezone manipulation
    cdef object _tz_localize(self, object series, object tz, object ambiguous, str nonexistent) except *
    cdef object _tz_convert(self, object series, object tz) except *
    cdef object _tz_switch(self, object series, object targ_tz, object base_tz, object ambiguous, str nonexistent, bint naive) except *
    # Frequency manipulation
    cdef pddt _round(self, str freq, object ambiguous, str nonexistent) except *
    cdef pddt _ceil(self, str freq, object ambiguous, str nonexistent) except *
    cdef pddt _floor(self, str freq, object ambiguous, str nonexistent) except *
    # Delta adjustment
    cdef pddt _delta(self, int years, int months, int days, int weeks, int hours, int minutes, int seconds, int microseconds) except *
    # Replace adjustment
    cdef pddt _replace(self, int year, int month, int day, int hour, int minute, int second, int microsecond, object tzinfo, int fold) except *
    # Between calculation
    cdef object _between(self, object other, str unit, bint inclusive) except *
    # Core methods
    cdef pddt _new(self, object series) except *
    cdef object _to_datetime(self, object timeobj) except *
    cdef object _parse_datetime(self, timeobj) except *
    cdef object _fill_default(self, object series) except *
    cdef object _np_to_series(self, np.ndarray array) except *
    cdef object _get_index(self) except *
    cdef object _get_naive(self) except *
    cdef int _parse_weekday(self, object weekday) except -1
    cdef int _parse_month(self, object month) except -1
    cdef str _parse_frequency(self, str freq) except *
    cdef _validate_am_non(self, object ambiguous, str nonexistent) except *
    cdef object _between_pddt(self, pddt pt, str unit, bint inclusive) except *
    cdef object _between_series(self, object series, str unit, bint inclusive) except *
    # Special methods
    cdef object _to_pddt(self, object other) except *
    cdef object _adj_other(self, object other) except *
    cdef pddt _copy(self) noexcept
