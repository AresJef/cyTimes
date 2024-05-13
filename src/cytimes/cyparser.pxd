# cython: language_level=3

from cpython cimport datetime
from libc.math cimport isfinite
from libc.stdlib cimport malloc, free, strtod, strtoll
from cpython.unicode cimport PyUnicode_GET_LENGTH as str_len
from cpython.unicode cimport PyUnicode_READ_CHAR as read_char
from cpython.unicode cimport PyUnicode_Substring as str_substr

# Constants
cdef:
    # . default config
    set CONFIG_PERTAIN, CONFIG_JUMP, CONFIG_UTC
    dict CONFIG_TZ, CONFIG_MONTH, CONFIG_WEEKDAY, CONFIG_HMS, CONFIG_AMPM
    # . datetime
    unsigned int[5] US_FRACTION_CORRECTION
    # . timezone
    set TIMEZONE_NAME_LOCAL
    
# Utils
cdef inline bint is_iso_sep(Py_UCS4 ch) except -1:
    """Check if is the separator `" "` or `"T"` for ISO format date & time  `<'bool'>`"""
    return ch == " " or ch == "t"

cdef inline bint is_isodate_sep(Py_UCS4 ch) except -1:
    """Check if is the separator `"-"` for ISO format date `<'bool'>`"""
    return ch == "-" or ch == "/"

cdef inline bint is_isoweek_sep(Py_UCS4 ch) except -1:
    """Check if is the separator `"W"` for ISO format week `<'bool'>`"""
    return ch == "w"

cdef inline bint is_isotime_sep(Py_UCS4 ch) except -1:
    """Check if is the separator `":"` for ISO format time `<'bool'>`"""
    return ch == ":"

cdef inline bint is_ascii_digit(Py_UCS4 ch) except -1:
    """Check if is ASCII digit number `<'bool'>`"""
    return "0" <= ch <= "9"
    
cdef inline bint is_ascii_alpha_upper(Py_UCS4 ch) except -1:
    """Check if is ASCII [A-Z] uppercase `<'bool'>`."""
    return "A" <= ch <= "Z"

cdef inline bint is_ascii_alpha_lower(Py_UCS4 ch) except -1:
    """Check if is ASCII [a-z] lowercase `<'bool'>`."""
    return "a" <= ch <= "z"

cdef inline bint is_ascii_alpha(Py_UCS4 ch) except -1:
    """Check if is ASCII [a-zA-Z] ignorecase `<'bool'>`."""
    return is_ascii_alpha_lower(ch) or is_ascii_alpha_upper(ch)

cdef inline unsigned int str_count(str s, str substr) except -1:
    """Count the number of occurrences of a sub string `<'int'>`.
    Equivalent to `s.count(substr)`."""
    return s.count(substr)

cdef inline double slice_to_float(str s, Py_ssize_t start, Py_ssize_t end):
    """Slice string from 'start' to 'end', and convert to `<'double'>`."""
    # Calculate the size of the token.
    cdef Py_ssize_t size = end - start
    # Allocate memory for the slice.
    cdef char* buffer = <char*>malloc(size + 1)
    if buffer == NULL:
        raise MemoryError("Failed to allocate memory for the string slice.")
    cdef Py_ssize_t i
    cdef Py_UCS4 ch
    try:
        # Assign the slice to the buffer.
        for i in range(size):
            ch = read_char(s, start + i)
            if not is_ascii_digit(ch) and ch not in ".,":
                raise ValueError("Invalid character to slice to float.")
            buffer[i] = ch
        # Null-terminate the buffer.
        buffer[size] = 0
        # Convert to double.
        return strtod(buffer, NULL)
    finally:
        # Free the memory.
        free(buffer)

cdef inline long long slice_to_int(str s, Py_ssize_t start, Py_ssize_t end):
    """Slice string from 'start' to 'end', and convert to `<'long long'>`."""
    # Calculate the size of the token.
    cdef Py_ssize_t size = end - start
    # Allocate memory for the slice.
    cdef char* buffer = <char*>malloc(size + 1)
    if buffer == NULL:
        raise MemoryError("Failed to allocate memory for the string slice.")
    cdef Py_ssize_t i
    cdef Py_UCS4 ch
    try:
        # Assign the slice to the buffer.
        for i in range(size):
            ch = read_char(s, start + i)
            if not is_ascii_digit(ch):
                raise ValueError("Invalid character to slice to integer.")
            buffer[i] = ch
        # Null-terminate the buffer.
        buffer[size] = 0
        # Convert to long long.
        return strtoll(buffer, NULL, 10)
    finally:
        # Free the memory.
        free(buffer)

cdef inline int token_to_int(str token):
    """Convert 'token' to `<'int'>`."""
    try:
        return int(token)
    except Exception as err:
        raise ValueError("Can't convert token '%s' to integer." % token) from err

cdef inline double token_to_float(str token):
    """Convert 'token' to `<'double'>`."""
    cdef double num
    try:
        num = float(token)
    except Exception as err:
        raise ValueError("Can't convert token '%s' to float." % token) from err
    if not isfinite(num):
        raise ValueError("Token '%s' does not represent a finite number." % token)
    return num

cdef inline unsigned int token_to_microsecond(str token, Py_ssize_t length) except -1:
    """Convert 'token' to microsecond fraction ('0001' or '000001'),
    and automatically adjust the microsecond value based on the
    token length `<'int'>`."""
    cdef unsigned int us
    # Validate fraction
    if length == 0:
        length = str_len(token)
    if length > 6:
        length = 6
        token = str_substr(token, 0, 6)
    elif length < 1:
        raise ValueError("Empty microsecond token: '%s'." % token)
    # Parse microsecond
    try:
        us = int(token)
    except Exception as err:
        raise ValueError("Invalid microsecond token: '%s'." % token) from err
    # Adjust fraction
    if length < 6:
        us *= US_FRACTION_CORRECTION[length - 1]
    return us

# Timelex
cdef list timelex(str dtstr, Py_ssize_t length=?)

# Result
cdef class Result:
    cdef:
        # Y/M/D
        int[3] _ymd
        int _idx, _yidx, _midx, _didx
        # Result
        int year, month, day, weekday
        int hour, minute, second, microsecond
        int ampm, tzoffset
        str tzname
        bint century_specified
    # Y/M/D
    cdef inline bint append(self, object value, unsigned int label) except -1
    cdef inline unsigned int populated(self) except -1
    cdef inline unsigned int labeled(self) except -1
    cdef inline bint could_be_day(self, int value) except -1
    # Result
    cdef inline bint prepare(self, bint day1st, bint year1st) except -1
    cdef inline bint valid(self) except -1

# Config
cdef class Config:
    cdef:
        # Settings
        bint _day1st, _year1st
        set _pertain, _jump, _utc
        dict _tz, _month, _weekday, _hms, _ampm
        # Keywords
        set _words
    # Validate
    cdef inline bint _construct_words(self) except -1
    cdef inline object _validate_word(self, str setting, object word)
    cdef inline object _validate_value(self, str setting, object value, int min, int max)

# Parser
cdef class Parser:
    cdef:
        # Config
        bint _day1st, _year1st, _ignoretz
        set _pertain, _jump, _utc
        dict _tz, _month, _weekday, _hms, _ampm
        # Result
        Result _result
        # Process
        unsigned int _iso_type
        list _tokens
        Py_ssize_t _token_count, _index
        str _token1, _token2, _token3, _token4
    # Parse
    cpdef datetime.datetime parse(self, str dtstr, object default=?, object day1st=?, object year1st=?, object ignoretz=?, object isoformat=?)
    # Process
    cdef inline bint _process(self, str dtstr, bint isoformat) except -1
    cdef inline bint _process_isoformat(self, str dtstr) except -1
    cdef inline bint _process_tokens(self, str dtstr) except -1
    # Build
    cdef inline datetime.datetime _build(self, str dtstr, object default)
    cdef inline datetime.datetime _build_datetime(self, str dtstr, object default, object tz)
    cdef inline datetime.datetime _handle_ambiguous_time(self, datetime.datetime dt, str tzname)
    # ISO format
    cdef inline bint _parse_iso_date(self, str dstr, Py_ssize_t length) except -1
    cdef inline bint _parse_iso_time(self, str tstr, Py_ssize_t length) except -1
    cdef inline bint _parse_iso_hms(self, str tstr, Py_ssize_t length) except -1
    cdef inline bint _parse_iso_extra(self, str estr, Py_ssize_t length, bint ignoretz) except -1
    cdef inline Py_ssize_t _find_iso_sep(self, str dtstr, Py_ssize_t length) except -1
    cdef inline Py_ssize_t _find_iso_utc(self, str tstr, Py_ssize_t length) except -1
    # Tokens
    cdef inline bint _parse_token_numeric(self, str token) except -1
    cdef inline bint _parse_token_month(self, str token) except -1
    cdef inline bint _parse_token_weekday(self, str token) except -1
    cdef inline bint _parse_token_hms(self, str token, double t_num) except -1
    cdef inline bint _parse_token_ampm(self, str token) except -1
    cdef inline bint _parse_token_tzname(self, str token) except -1
    cdef inline bint _parse_token_tzoffset(self, str token) except -1
    # Tokens utils
    cdef inline bint _set_hms(self, str token, double t_num, int hms) except -1
    cdef inline bint _set_hm(self, double t_num) except -1
    cdef inline bint _set_ms(self, double t_num) except -1
    cdef inline bint _set_sf(self, str token) except -1
    cdef inline unsigned int _adjust_ampm(self, int hour, int ampm) except -1
    cdef inline unsigned int _could_be_tzname(self, str token) except -1
    # Tokens access
    cdef inline bint _parse_tokens(self, str dtstr, Py_ssize_t length) except -1
    cdef inline str _access_token(self, Py_ssize_t index)
    cdef inline str _access_token1(self)
    cdef inline str _access_token2(self)
    cdef inline str _access_token3(self)
    cdef inline str _access_token4(self)
    cdef inline bint _reset_tokens(self) except -1
    # Config
    cdef inline bint _is_token_pertain(self, object token) except -1
    cdef inline bint _is_token_jump(self, object token) except -1
    cdef inline bint _is_token_utc(self, object token) except -1
    cdef inline int _token_to_month(self, object token) except -2
    cdef inline int _token_to_weekday(self, object token) except -2
    cdef inline int _token_to_hms(self, object token) except -2
    cdef inline int _token_to_ampm(self, object token) except -2
    cdef inline int _token_to_tzoffset(self, object token) except -100001
