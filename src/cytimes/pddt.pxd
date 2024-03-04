# cython: language_level=3

cimport numpy as np
from cpython cimport datetime

# Constants
cdef:
    # . calendar
    np.ndarray DAYS_BR_QUARTER_NDARRAY
    # . time
    datetime.time TIME_START, TIME_END
    # . unit
    set UNIT_FREQUENCY
    # . offset
    object OFST_DATEOFFSET, OFST_MICRO, OFST_DAY
    object OFST_MONTHEND, OFST_MONTHBEGIN
    object OFST_QUARTEREND, OFST_QUARTERBEGIN
    object OFST_YEAREND, OFST_YEARBEGIN
    # . type
    object TP_SERIES, TP_DATETIME64, TP_DATETIMEINDEX
    object TP_TIMEDELTA, TP_TIMEDELTAINDEX, TP_BASEOFFSET
    # . function
    object FN_PD_TODATETIME, FN_PD_TOTIMEDELTA
    object FN_NP_ABS, FN_NP_FULL, FN_NP_WHERE, FN_NP_MINIMUM

# pddt (Pandas Datetime)
cdef class pddt:
    cdef:
        # . config
        object _default
        str _format
        bint _day1st, _year1st, _utc, _exact
        # . datetime
        object _dts, _dts_index
        bint _is_unit_ns
        unsigned int _dts_len
        str _dts_unit
        np.ndarray _dts_value
        object _dts_naive, _series_index
    # Manipulate: Timezone
    cdef bint _validate_ambiguous_nonexistent(pddt self, object ambiguous, object nonexistent) except -1
    # Core methods
    cdef pddt _new(pddt self, object s) except *
    cdef object _parse_dtobj(pddt self, object dtobj) except *
    cdef object _parse_dtseries(pddt self, object dtobj) except *
    cdef datetime.datetime _parse_datetime(pddt self, object dtobj, object default) except *
    cdef unsigned int _parse_month(pddt self, object month) except -1
    cdef unsigned int _parse_weekday(pddt self, object month) except -1
    cdef object _parse_frequency(pddt self, object freq) except *
    cdef bint _get_is_unit_ns(pddt self) except -1
    cdef unsigned int _get_dts_len(pddt self) except -1
    cdef str _get_dts_unit(pddt self) noexcept
    cdef np.ndarray _get_dts_value(pddt self) noexcept
    cdef object _get_dts_naive(pddt self) noexcept
    cdef object _get_series_index(pddt self) noexcept
    cdef object _gen_timedelta(pddt self, object arg, object unit) except *
    cdef object _array_to_series(pddt self, object arr) except *
    cdef object _convert_pddt(pddt self, object other) except *
    cdef object _convert_other(pddt self, object other) except *
