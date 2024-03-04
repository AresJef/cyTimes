# cython: language_level=3

from cpython cimport datetime

# Constants
cdef:
# . default config
    set CONFIG_PERTAIN, CONFIG_JUMP, CONFIG_UTC
    dict CONFIG_MONTH, CONFIG_WEEKDAY, CONFIG_HMS
    dict CONFIG_AMPM, CONFIG_TZINFO
    # . datetime
    int[5] US_FRACTION_CORRECTION
    # . timezone
    set TIMEZONE_NAME_LOCAL
    # . charactor
    Py_UCS4 CHAR_NULL, CHAR_SPACE, CHAR_PLUS, CHAR_COMMA, CHAR_DASH, CHAR_PERIOD
    Py_UCS4 CHAR_SLASH, CHAR_COLON, CHAR_LOWER_T, CHAR_LOWER_W, CHAR_LOWER_Z
    # . type
    object TP_PARSERINFO
    
# ISO Format
cdef bint is_iso_sep(Py_UCS4 char) except -1
cdef bint is_isodate_sep(Py_UCS4 char) except -1
cdef bint is_isoweek_sep(Py_UCS4 char) except -1
cdef bint is_isotime_sep(Py_UCS4 char) except -1
cdef unsigned int parse_us_fraction(str us_frac_str, unsigned int us_frac_len=?) except -1

# Timelex
cdef bint is_ascii_digit(Py_UCS4 char) except -1
cdef bint is_ascii_alpha(Py_UCS4 char) except -1
cdef bint is_ascii_alpha_lower(Py_UCS4 char) except -1
cdef bint is_ascii_alpha_upper(Py_UCS4 char) except -1

cdef unsigned int str_count(str s, str char) except -1
cdef list parse_timelex(str dtstr, unsigned int length=?) except *

# Result
cdef class Result:
    cdef:
        # YMD
        int[3] _ymd
        int _ymd_idx, _ymd_yidx, _ymd_midx, _ymd_didx
        # Result
        int year, month, day, weekday
        int hour, minute, second, microsecond
        int ampm, tzoffset
        str tzname
        bint _century_specified
    # YMD
    cdef bint append_ymd(Result self, object value, unsigned int label) except -1
    cdef unsigned int ymd_values(Result self) except -1
    cdef bint could_be_day(Result self, int value) except -1
    cdef bint _set_ymd(Result self, int value) except -1
    cdef unsigned int _labeled_ymd(Result self) except -1
    cdef bint _resolve_ymd(Result self, bint day1st, bint year1st) except -1
    # Result
    cdef bint prepare(Result self, bint day1st, bint year1st) except -1
    cdef bint is_valid(Result self) except -1

# Config
cdef class Config:
    cdef:
        # Settings
        bint _day1st, _year1st
        set _pertain, _jump, _utc
        dict _month, _weekday, _hms
        dict _ampm, _tzinfo
        # Keywords
        set _keywords
    # Validate
    cdef bint _construct_keywords(Config self) except -1
    cdef object _validate_keyword(Config self, str setting, object word) except *
    cdef object _validate_value(Config self, str setting, object value, int min, int max) except *

# Parser
cdef class Parser:
    cdef:
        # Config
        bint _day1st, _year1st, _ignoretz, _fuzzy
        set _pertain, _jump, _utc
        dict _month, _weekday, _hms, _ampm, _tzinfo
        # Result
        Result _result
        # Process
        str _dtstr
        unsigned int _dtstr_len, _isodate_type
        list _tokens
        int _tokens_count, _index
        str _token_r1, _token_r2, _token_r3, _token_r4
    # Parsing
    cpdef datetime.datetime parse(
        Parser self, str dtstr, object default=?, object day1st=?, 
        object year1st=?, object ignoretz=?, object fuzzy=?) except *
    cdef bint _process_iso(Parser self) except -1
    cdef bint _process_core(Parser self) except -1
    cdef datetime.datetime _build(Parser self, object default) except *
    cdef datetime.datetime _build_datetime(Parser self, object default, object tzinfo) except *
    cdef datetime.datetime _handle_ambiguous_time(Parser self, datetime.datetime dt, str tzname) except *
    # ISO format
    cdef unsigned int _find_isoformat_sep(Parser self) except -1
    cdef bint _parse_isoformat_date(Parser self, str dstr, unsigned int length) except -1
    cdef bint _parse_isoformat_time(Parser self, str tstr, unsigned int length) except -1
    cdef bint _parse_isoformat_hms(Parser self, str tstr, unsigned int length) except -1
    cdef unsigned int _find_isoformat_tz(Parser self, str tstr, unsigned int length) except -1
    cdef Py_UCS4 _get_char(Parser self, unsigned int index) noexcept
    # Numeric token
    cdef bint _parse_numeric_token(Parser self, str token) except -1
    cdef double _covnert_numeric_token(Parser self, str token) except *
    # Month token
    cdef bint _parse_month_token(Parser self, str token) except -1
    # Weekday token
    cdef bint _parse_weekday_token(Parser self, str token) except -1
    # HMS token
    cdef bint _parse_hms_token(Parser self, str token, double value) except -1
    cdef bint _set_hms_result(Parser self, str token, double value, int hms) except -1
    cdef bint _set_hour_and_minite(Parser self, double value) except -1
    cdef bint _set_minite_and_second(Parser self, double value) except -1
    cdef bint _set_second_and_us(Parser self, str token) except -1
    # AM/PM token
    cdef bint _parse_ampm_token(Parser self, str token) except -1
    cdef unsigned int _adjust_ampm_hour(Parser self, int hour, int ampm) except -1
    # Tzname token
    cdef bint _parse_tzname_token(Parser self, str token) except -1
    cdef unsigned int _could_be_tzname(Parser self, str token) except -1
    # Tzoffset token
    cdef bint _parse_tzoffset_token(Parser self, str token) except -1
    cdef int _calculate_tzoffset(Parser self, str token_r1, int sign, int offset) noexcept
    # Get token
    cdef str _get_token(Parser self, int index) noexcept
    cdef str _get_token_r1(Parser self) noexcept
    cdef str _get_token_r2(Parser self) noexcept
    cdef str _get_token_r3(Parser self) noexcept
    cdef str _get_token_r4(Parser self) noexcept
    # Config
    cdef bint _is_token_pertain(Parser self, object token) except -1
    cdef bint _is_token_jump(Parser self, object token) except -1
    cdef bint _is_token_utc(Parser self, object token) except -1
    cdef int _token_to_month(Parser self, object token) noexcept
    cdef int _token_to_weekday(Parser self, object token) noexcept
    cdef int _token_to_hms(Parser self, object token) noexcept
    cdef int _token_to_ampm(Parser self, object token) noexcept
    cdef int _token_to_tzoffset(Parser self, object token) noexcept
