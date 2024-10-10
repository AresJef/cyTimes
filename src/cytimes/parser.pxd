# cython: language_level=3

from cpython cimport datetime

# Constants
cdef:
    set CONFIG_PERTAIN 
    set CONFIG_JUMP
    set CONFIG_UTC
    dict CONFIG_TZ
    dict CONFIG_MONTH
    dict CONFIG_WEEKDAY
    dict CONFIG_HMS_FLAG
    dict CONFIG_AMPM_FLAG

# Timelex
cpdef list timelex(str dtstr)
cdef list _timelex(str dtstr, Py_ssize_t start=?, Py_ssize_t size=?)

# Configs
cdef class Configs:
    cdef:
        # . settings
        bint _year1st
        bint _day1st
        set _pertain
        set _jump
        set _utc
        dict _tz
        dict _month
        dict _weekday
        dict _hms_flag
        dict _ampm_flag
        # . internal
        set _words
    # Internal
    cdef inline bint _construct(self) except -1
    cdef inline str _validate_word(self, str settings, object word)
    cdef inline object _validate_value(self, str settings, object value, int min, int max)
    cdef inline object _validate_value_month(self, object value)
    cdef inline object _validate_value_weekday(self, object value)
    cdef inline object _validate_value_hms_flag(self, object value)
    cdef inline object _validate_value_ampm_flag(self, object value)
    cdef inline object _validate_value_tzoffset(self, object value)

# Parser
cdef class Result:
    cdef:
        # . Y/M/D
        int _ymd[3]
        int _idx
        int _yidx
        int _midx
        int _didx
        # . values
        int year
        int month
        int day
        int weekday
        int hour
        int minute
        int second
        int microsecond
        int ampm
        int tzoffset
        bint century_specified
    # Y/M/D
    cdef inline bint set_ymd_int(self, long long value, int label) except -1
    cdef inline bint set_ymd_str(self, str value, int label) except -1
    cdef inline bint _set_ymd(self, int value, int label) except -1
    cdef inline int ymd_populated(self) except -1
    cdef inline int ymd_labeled(self) except -1
    cdef inline bint could_be_day(self, long long value) except -1
    # Values
    cdef inline bint prepare(self, bint year1st, bint day1st) except -1
    cdef inline bint valid(self) except -1
    cdef inline bint reset(self) except -1

cdef class Parser:
    cdef:
        # . settings
        bint _ignoretz
        # . configs
        bint _year1st
        bint _day1st
        set _pertain
        set _jump
        set _utc
        dict _tz
        dict _month
        dict _weekday
        dict _hms_flag
        dict _ampm_flag
        # Result
        Result _res
        # Process
        Py_ssize_t _idx
        Py_ssize_t _size
        list _tokens
        str _tk1, _tk2, _tk3, _tk4
    # Parse
    cpdef datetime.datetime parse(
        self, str dtstr, object default=?, 
        object year1st=?, object day1st=?, 
        bint ignoretz=?, bint isoformat=?)
    cdef inline bint _process(self, str dtstr, bint isoformat) except -1
    # Build
    cdef inline datetime.datetime _build(self, str dtstr, object default)
    cdef inline datetime.datetime _generate_dt(self, object default, object tz)
    # ISO format
    cdef inline bint _process_iso_format(self, str dtstr) except -1
    # . parser
    cdef inline bint _parse_iso_date(self, str dtstr) except -1
    cdef inline bint _parse_iso_time(self, str dtstr) except -1
    cdef inline bint _parse_iso_extra(self, str dtstr) except -1
    # Timelex tokens
    cdef inline bint _process_timelex_tokens(self, str dtstr) except -1
    # . parser
    cdef inline bint _parse_token_numeric(self, str token) except -1
    cdef inline bint _parse_token_month(self, str token) except -1
    cdef inline bint _parse_token_weekday(self, str token) except -1
    cdef inline bint _parse_token_hmsf(self, double t_val) except -1
    cdef inline bint _parse_token_ampm_flag(self, str token) except -1
    cdef inline bint _parse_token_tzname(self, str token) except -1
    cdef inline bint _parse_token_tzoffset(self, str token) except -1
    # . tokens
    cdef inline str _token(self, Py_ssize_t idx)
    cdef inline str _token1(self)
    cdef inline str _token2(self)
    cdef inline str _token3(self)
    cdef inline str _token4(self)
    cdef inline bint _reset_tokens(self) except -1
    # . utils
    cdef inline bint _set_hmsf(self, double t_val, int hms_flag) except -1
    cdef inline bint _set_hhmm(self, double t_val) except -1
    cdef inline bint _set_mmss(self, double t_val) except -1
    cdef inline bint _set_ssff(self, double t_val) except -1
    cdef inline int _adj_hour_ampm(self, int hour, int ampm_flag) except -1
    # . configs
    cdef inline bint _is_token_pertain(self, object token) except -1
    cdef inline bint _is_token_jump(self, object token) except -1
    cdef inline bint _is_token_utc(self, object token) except -1
    cdef inline int _token_to_month(self, object token) except -2
    cdef inline int _token_to_weekday(self, object token) except -2
    cdef inline int _token_to_tzoffset(self, object token) except -200_000
    cdef inline int _token_to_hms_flag(self, object token) except -2
    cdef inline int _token_to_ampm_flag(self, object token) except -2

cdef Parser _default_parser

cpdef datetime.datetime parse(
    str dtstr, object default=?, 
    object year1st=?, object day1st=?, 
    bint ignoretz=?, bint isoformat=?, 
    Configs cfg=?)

cpdef datetime.datetime parse_dtobj(
    object dtobj, object default=?, 
    object year1st=?, object day1st=?, 
    bint ignoretz=?, bint isoformat=?, 
    Configs cfg=?)
